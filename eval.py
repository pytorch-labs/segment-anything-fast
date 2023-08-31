import collections
import time

# for now, speed up point calculation with diskcache
# depending on requirements of our eval, this may have to change in the future
import diskcache
import fire
import pandas as pd
import tqdm
import os
import sys

import torch
from pycocotools.coco import COCO
import numpy as np
from scipy import ndimage
import skimage.io as io
import skimage.color as color

def _get_center_point(mask, ann_id, cache):
    """
    This is a rudimentary version of https://arxiv.org/pdf/2304.02643.pdf,
    section D.1.Point Sampling

    From the paper: "The first point is chosen deterministically as the point
    farthest from the object boundary."

    The code below is an approximation of this.

    First, we try to calculate the center of mass. If it's inside the mask, we
    stop here.

    The centroid may be outside of the mask for some mask shapes. In this case
    we do a slow hack, specifically, we check for the
    minumum of the maximum distance from the boundary in four directions
    (up, right, down, left), and take the point with the maximum of these
    minimums. Note: this is not performant for large masks.

    Returns the center point in (x, y) format
    """
    if ann_id in cache:
        return cache[ann_id]

    # try the center of mass, keep it if it's inside the mask
    com_y, com_x = ndimage.center_of_mass(mask)
    com_y, com_x = int(round(com_y, 0)), int(round(com_x, 0))
    if mask[com_y][com_x]:
        cache[ann_id] = (com_x, com_y)
        return (com_x, com_y)

    # if center of mass didn't work, do the slow manual approximation

    # up, right, down, left
    # TODO(future): approximate better by adding more directions
    distances_to_check_deg = [0, 90, 180, 270]

    global_min_max_distance = float('-inf')
    global_coords = None
    # For now, terminate early to speed up the calculation as long as
    # the point sample is gooe enough. This sacrifices the quality of point
    # sampling for speed. In the future we can make this more accurate.
    DISTANCE_GOOD_ENOUGH_THRESHOLD = 20

    # Note: precalculating the bounding box could be somewhat
    #   helpful, but checked the performance gain and it's not much
    #   so leaving it out to keep the code simple.
    # Note: tried binary search instead of incrementing by one to
    #   travel up/right/left/down, but that does not handle masks
    #   with all shapes properly (there could be multiple boundaries).
    for row_idx in range(mask.shape[0]):
        for col_idx in range (mask.shape[1]):
            cur_point = mask[row_idx, col_idx]

            # skip points inside bounding box but outside mask
            if not cur_point:
                continue

            max_distances = []
            for direction in distances_to_check_deg:
                # TODO(future) binary search instead of brute forcing it if we
                # need a speedup, with the cache it doesn't really matter though
                if direction == 0:
                    # UP
                    cur_row_idx = row_idx

                    while cur_row_idx >= 0 and mask[cur_row_idx, col_idx]:
                        cur_row_idx = cur_row_idx - 1
                    cur_row_idx += 1
                    distance = row_idx - cur_row_idx
                    max_distances.append(distance)

                elif direction == 90:
                    # RIGHT
                    cur_col_idx = col_idx

                    while cur_col_idx <= mask.shape[1] - 1 and \
                            mask[row_idx, cur_col_idx]:
                        cur_col_idx += 1
                    cur_col_idx -= 1
                    distance = cur_col_idx - col_idx
                    max_distances.append(distance)

                elif direction == 180:
                    # DOWN
                    cur_row_idx = row_idx
                    while cur_row_idx <= mask.shape[0] - 1 and \
                            mask[cur_row_idx, col_idx]:
                        cur_row_idx = cur_row_idx + 1
                    cur_row_idx -= 1
                    distance = cur_row_idx - row_idx
                    max_distances.append(distance)

                elif direction == 270:
                    # LEFT
                    cur_col_idx = col_idx
                    while cur_col_idx >= 0 and mask[row_idx, cur_col_idx]:
                        cur_col_idx -= 1
                    cur_col_idx += 1
                    distance = col_idx - cur_col_idx
                    max_distances.append(distance)

            min_max_distance = min(max_distances)
            if min_max_distance > global_min_max_distance:
                global_min_max_distance = min_max_distance
                global_coords = (col_idx, row_idx)
            if global_min_max_distance >= DISTANCE_GOOD_ENOUGH_THRESHOLD:
                break

    cache[ann_id] = global_coords
    return global_coords

def _iou(mask1, mask2):
    assert mask1.dim() == 3
    assert mask2.dim() == 3
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return (intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2)))

def build_datapoint(imgId, coco, pixel_mean, pixel_std, coco_root_dir, coco_slice_name, catIds, cache, predictor):
    img = coco.loadImgs(imgId)[0]

    file_location = f'{coco_root_dir}/{coco_slice_name}/{img["file_name"]}'
    I = io.imread(file_location)
    if len(I.shape) == 2:
        # some images, like img_id==61418, are grayscale
        # convert to RGB to ensure the rest of the pipeline works
        I = color.gray2rgb(I)

    # load and display instance annotations
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # approximate the center point of each mask
    coords_list = []
    gt_masks_list = []
    for ann in anns:
        ann_id = ann['id']
        mask = coco.annToMask(ann)
        gt_masks_list.append(torch.tensor(mask))
        coords = _get_center_point(mask, ann_id, cache)
        coords_list.append(coords)


    image = I

    # predictor_set_image begin
    # Transform the image to the form expected by the model
    input_image = predictor.transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    predictor_input_size = input_image_torch.shape[-2:]


    # Preprocess
    x = input_image_torch
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = predictor.model.image_encoder.img_size - h
    padw = predictor.model.image_encoder.img_size - w
    x = torch.nn.functional.pad(x, (0, padw, 0, padh))

    gt_masks_list = torch.stack(gt_masks_list) if len(gt_masks_list) else None
    return image, coords_list, gt_masks_list, anns, x, predictor_input_size

def build_data(coco_img_ids, coco, catIds, coco_root_dir, coco_slice_name, cache, predictor, use_half, use_half_decoder):

    pixel_mean = predictor.model.pixel_mean.cpu()
    pixel_std = predictor.model.pixel_std.cpu()

    def build_batch(indicies):
        batch = [[], [], [], [], [], [], [], [], [], [], []]
        batch[8] = [0]
        batch[10] = [0]
        for img_idx in indicies:
            imgId = coco_img_ids[img_idx]

            I, coords_list, gt_masks_list, anns, x, predictor_input_size = build_datapoint(imgId,
                                                                                           coco,
                                                                                           pixel_mean,
                                                                                           pixel_std,
                                                                                           coco_root_dir,
                                                                                           coco_slice_name,
                                                                                           catIds,
                                                                                           cache,
                                                                                           predictor)
            if len(coords_list) == 0:
                continue
            batch[0].append(I)
            coords_list = predictor.transform.apply_coords(np.array(coords_list), I.shape[:2])
            coords_list = torch.tensor(coords_list, dtype=torch.float)
            batch[1].append(coords_list.reshape(-1))
            batch[2].append(coords_list.size())
            batch[3].append(gt_masks_list.reshape(-1))
            batch[4].append(anns)
            batch[5].append(x)
            batch[6].append(predictor_input_size)
            batch[7].append(img_idx)
            batch[8].append(coords_list.numel() + batch[8][-1])
            batch[9].append(gt_masks_list.size())
            batch[10].append(gt_masks_list.numel() + batch[10][-1])
        if use_half_decoder:
            batch[1] = torch.cat(batch[1]).half() if len(batch[0]) > 0 else None
        else:
            batch[1] = torch.cat(batch[1]) if len(batch[0]) > 0 else None
        batch[3] = torch.cat(batch[3]) if len(batch[0]) > 0 else None
        if use_half:
            batch[5] = torch.cat(batch[5]).half() if len(batch[0]) > 0 else None
        else:
            batch[5] = torch.cat(batch[5]) if len(batch[0]) > 0 else None
        return batch

    return build_batch


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
        input_pointss = coords_lists.to(device=predictor.device, non_blocking=True)
        gt_masks_lists = gt_masks_lists.to(device=predictor.device, non_blocking=True)

        with torch.no_grad():
            features_batch = encoder(input_image_batch)

        for batch_idx in range(len(Is)):
            features = features_batch.narrow(0, batch_idx, 1)
            predictor_input_size = predictor_input_sizes[batch_idx]
            image = Is[batch_idx]
            img_idx = img_idxs[batch_idx]

            predictor.reset_image()
            predictor.original_size = image.shape[:2]
            predictor.input_size = predictor_input_size # tuple(input_image.shape[-2:])
            predictor.features = features
            predictor.is_image_set = True

            anns = annss[batch_idx]

            input_points = input_pointss[coords_offsets[batch_idx]:coords_offsets[batch_idx+1]].view(coords_lists_sizes[batch_idx])
            gt_masks_list = gt_masks_lists[gt_masks_offsets[batch_idx]:gt_masks_offsets[batch_idx+1]].view(gt_masks_sizes[batch_idx])

            input_points = input_points.unsqueeze(1)
            num_points = len(input_points)

            @torch.no_grad()
            def create_top_score_ious(input_points, num_points):
                fg_labels = torch.ones((num_points, 1), dtype=torch.int, device=predictor.device)

                # TODO: Break this up further to batch more computation.
                masks, scores, logits = predictor.predict_torch(
                    point_coords=input_points,
                    point_labels=fg_labels,
                    multimask_output=True,
                )

                argmax_scores = torch.argmax(scores, dim=1)
                inference_masks = masks.gather(1, argmax_scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((masks.size(0), 1, masks.size(2), masks.size(3)))).squeeze(1)
                top_score_ious = _iou(inference_masks, gt_masks_list)
                return top_score_ious, inference_masks

            if compile_create_top_score_ious:
                top_score_ious, inference_masks = (torch.compile(create_top_score_ious))(input_points, num_points)
            else:
                top_score_ious, inference_masks = create_top_score_ious(input_points, num_points)
            for idx in range(num_points):
                results.append(
                    [img_idx, anns[idx]['id'], anns[idx]['category_id'], top_score_ious[idx]])

            # # TODO(future): clean this up, ideally we should save in COCO format
            if save_inference_masks:
                torch.save(inference_masks, f'{mask_debug_out_dir}/{img_idx}_masks.pth')
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

    annFile='{}/annotations/instances_{}.json'.format(coco_root_dir, coco_slice_name)

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_cat = {cat['id']: cat for cat in cats}
    nms=[cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    if coco_category_names is not None:
        catIds = coco.getCatIds(catNms=coco_category_names)
    else:
        catIds = coco.getCatIds();
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
