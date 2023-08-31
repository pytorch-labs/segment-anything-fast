# for now, speed up point calculation with diskcache
# depending on requirements of our eval, this may have to change in the future
import diskcache
import pandas as pd
import tqdm

import torch
from pycocotools.coco import COCO
from scipy import ndimage


def _iou(mask1, mask2):
    assert mask1.dim() == 3
    assert mask2.dim() == 3
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return (intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2)))


def build_results(batched_data_iter, predictor, mask_debug_out_dir, batch_size, save_inference_masks, time_per_batch, compile_create_top_score_ious):

    bidx = 0
    results = []
    encoder = predictor.model.image_encoder
    batch_ms = []
    for Is, coords_lists, coords_lists_sizes, gt_masks_lists, annss, xs, predictor_input_sizes, img_idxs, coords_offsets, gt_masks_sizes, gt_masks_offsets in tqdm.tqdm(batched_data_iter):
        if coords_lists is None:
            continue
        if time_per_batch:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        input_image_batch = xs.to(device=predictor.device, non_blocking=True)
        input_pointss = coords_lists.to(
            device=predictor.device, non_blocking=True)
        gt_masks_lists = gt_masks_lists.to(
            device=predictor.device, non_blocking=True)

        with torch.no_grad():
            features_batch = encoder(input_image_batch)

        for batch_idx in range(len(Is)):
            features = features_batch.narrow(0, batch_idx, 1)
            predictor_input_size = predictor_input_sizes[batch_idx]
            image = Is[batch_idx]
            img_idx = img_idxs[batch_idx]

            predictor.reset_image()
            predictor.original_size = image.shape[:2]
            # tuple(input_image.shape[-2:])
            predictor.input_size = predictor_input_size
            predictor.features = features
            predictor.is_image_set = True

            anns = annss[batch_idx]

            input_points = input_pointss[coords_offsets[batch_idx]                                         :coords_offsets[batch_idx+1]].view(coords_lists_sizes[batch_idx])
            gt_masks_list = gt_masks_lists[gt_masks_offsets[batch_idx]                                           :gt_masks_offsets[batch_idx+1]].view(gt_masks_sizes[batch_idx])

            input_points = input_points.unsqueeze(1)
            num_points = len(input_points)

            @torch.no_grad()
            def create_top_score_ious(input_points, num_points):
                fg_labels = torch.ones(
                    (num_points, 1), dtype=torch.int, device=predictor.device)

                # TODO: Break this up further to batch more computation.
                masks, scores, logits = predictor.predict_torch(
                    point_coords=input_points,
                    point_labels=fg_labels,
                    multimask_output=True,
                )

                argmax_scores = torch.argmax(scores, dim=1)
                inference_masks = masks.gather(1, argmax_scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    (masks.size(0), 1, masks.size(2), masks.size(3)))).squeeze(1)
                top_score_ious = _iou(inference_masks, gt_masks_list)
                return top_score_ious, inference_masks

            if compile_create_top_score_ious:
                top_score_ious, inference_masks = (torch.compile(
                    create_top_score_ious))(input_points, num_points)
            else:
                top_score_ious, inference_masks = create_top_score_ious(
                    input_points, num_points)
            for idx in range(num_points):
                results.append(
                    [img_idx, anns[idx]['id'], anns[idx]['category_id'], top_score_ious[idx]])

            # # TODO(future): clean this up, ideally we should save in COCO format
            if save_inference_masks:
                torch.save(inference_masks,
                           f'{mask_debug_out_dir}/{img_idx}_masks.pth')
        if time_per_batch:
            end_event.record()
            torch.cuda.synchronize()
            batch_ms.append(start_event.elapsed_time(end_event))

    if time_per_batch:
        return results, batch_ms
    return results


def do_eval(
    predictor,
    coco_root_dir,
    coco_slice_name,
    coco_category_names,
    point_sampling_cache_dir,
    mask_debug_out_dir,
    limit,
    img_id,
    silent=False,
    batch_size=1,
    save_inference_masks=True,
    report_batch_timings=False,
    num_workers=0,
    use_half=False,
    use_half_decoder=False,
    compile_create_top_score_ious=False,
):
    cache = diskcache.Cache(point_sampling_cache_dir)
    # make sure you clear the cache if you change the point sampling algorithm
    # cache.clear()

    annFile = '{}/annotations/instances_{}.json'.format(
        coco_root_dir, coco_slice_name)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_cat = {cat['id']: cat for cat in cats}
    nms = [cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    if coco_category_names is not None:
        catIds = coco.getCatIds(catNms=coco_category_names)
    else:
        catIds = coco.getCatIds()
    if not silent:
        print('catIds', catIds)

    if img_id is not None:
        coco_img_ids = [img_id]
    elif coco_category_names is None:
        coco_img_ids = coco.getImgIds()
    else:
        coco_img_ids = coco.getImgIds(catIds=catIds)

    total_images = len(coco_img_ids)
    if not silent:
        print('total images', len(coco_img_ids))

    from data import build_data
    build_batch = build_data(coco_img_ids,
                             coco,
                             catIds,
                             coco_root_dir,
                             coco_slice_name,
                             cache,
                             predictor,
                             use_half,
                             use_half_decoder)

    limit = len(coco_img_ids) if limit is None else limit
    batched_data_iter = torch.utils.data.DataLoader(list(range(limit)),
                                                    batch_size=batch_size,
                                                    collate_fn=build_batch,
                                                    num_workers=num_workers,
                                                    pin_memory=True)

    results = build_results(batched_data_iter,
                            predictor,
                            mask_debug_out_dir,
                            batch_size,
                            save_inference_masks,
                            report_batch_timings,
                            compile_create_top_score_ious)
    if report_batch_timings:
        results, batch_ms = results
        batch_ms = torch.tensor(batch_ms)
        batch_ms_min = batch_ms.min().item()
        batch_ms_max = batch_ms.min().item()
        if not silent:
            print("batch_ms_min: ", batch_ms_min)
            print("batch_ms_max: ", batch_ms_max)

    # Avoid CUDA sync by deferring item call.
    results = [[r[0], r[1], r[2], r[3].item()] for r in results]

    df = pd.DataFrame(results, columns=['img_id', 'ann_id', 'cat_id', 'iou'])
    df.to_csv(f'{mask_debug_out_dir}/df.csv')
    df['supercategory'] = df['cat_id'].map(
        lambda cat_id: cat_id_to_cat[cat_id]['supercategory'])
    df['category'] = df['cat_id'].map(
        lambda cat_id: cat_id_to_cat[cat_id]['name'])

    # TODO: cross reference the specifics of how we calculate mIoU with
    # the SAM folks (should it be per dataset, per category, per image, etc)
    # currently, just calculate them all

    # TODO: QOL save the summaries to file

    # per category
    per_category = pd.pivot_table(
        df, values='iou', index=['cat_id', 'supercategory', 'category'],
        aggfunc=('mean', 'count'))
    if not silent:
        print('\nmIoU averaged per category')
        print(per_category)

    # per super-category
    per_supercategory = pd.pivot_table(
        df, values='iou', index=['supercategory'],
        aggfunc=('mean', 'count'))
    if not silent:
        print('\nmIoU averaged per supercategory')
        print(per_supercategory)

    # per all selected masks
    per_all_masks_agg = df['iou'].agg(['mean', 'count'])
    if not silent:
        print('\nmIoU averaged per all selected masks')
        print(per_all_masks_agg)
    metrics = {'catIds': catIds,
               'total_images': total_images,
               'results_df': df,
               'batch_size': batch_size}
    if report_batch_timings:
        metrics['batch_ms'] = batch_ms
    return metrics
