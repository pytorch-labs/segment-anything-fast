# for now, speed up point calculation with diskcache
# depending on requirements of our eval, this may have to change in the future
import diskcache
import tqdm

import torch
import fire
from metrics import calculate_miou, create_result_entry
from data import build_data, setup_coco_img_ids
from quant import apply_dynamic_quant

from segment_anything import sam_model_registry, SamPredictor

torch._dynamo.config.cache_size_limit = 5000

def unbind_jagged(data, offsets, sizes):
    return [data[offsets[batch_idx]:offsets[batch_idx+1]].view(sizes[batch_idx]) for batch_idx in range(len(sizes))]


def build_results(batched_data_iter,
                  predictor,
                  mask_debug_out_dir,
                  batch_size,
                  compile_create_top_score_ious):

    results = []
    encoder = predictor.model.image_encoder
    batch_ms = []
    for Is, coords_lists, coords_lists_sizes, gt_masks_lists, annss, xs, predictor_input_sizes, img_idxs, coords_offsets, gt_masks_sizes, gt_masks_offsets in tqdm.tqdm(batched_data_iter):
        if coords_lists is None:
            continue

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

        input_pointss = unbind_jagged(input_pointss, coords_offsets, coords_lists_sizes)
        gt_masks_lists = unbind_jagged(gt_masks_lists, gt_masks_offsets, gt_masks_sizes)

        for batch_idx, (input_points, gt_masks_list) in enumerate(zip(input_pointss, gt_masks_lists)):
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

            input_points = input_points.unsqueeze(1)
            num_points = len(input_points)

            fg_labels = torch.ones(
                (num_points, 1), dtype=torch.int, device=predictor.device)

            # TODO: Break this up further to batch more computation.
            masks, scores, logits = predictor.predict_torch(
                point_coords=input_points,
                point_labels=fg_labels,
                multimask_output=True,
            )
            results += create_result_entry(annss[batch_idx], gt_masks_list, masks, scores, img_idx)


        end_event.record()
        torch.cuda.synchronize()
        batch_ms.append(start_event.elapsed_time(end_event))


    return results, torch.tensor(batch_ms)


def run_do_eval_with_profile(*args, **kwargs):
    with torch.profiler.profile(
            # activities=[torch.profiler.ProfilerActivity.CPU,
            #             torch.profiler.ProfilerActivity.CUDA],
            with_stack=True,
            profile_memory=True,
            record_shapes=True) as prof:
        result = do_eval(*args, **kwargs)
    # prof.export_chrome_trace("/tmp/example_trace_cupti.json.gz")
    print("0")
    from torch.cuda._memory_viz import profile_plot
    print("1")
    with open('/tmp/output_cpuhrsch.html', 'w') as f:
        print("20")
        ppp = profile_plot(prof)
        print("21")
        f.write(ppp)
    print("3")
    return result


def run(
    coco_root_dir,
    coco_slice_name,
    sam_checkpoint_base_path,
    sam_model_type,
    point_sampling_cache_dir,
    mask_debug_out_dir,
    batch_size=1,
    print_header=False,
    coco_category_names=None,
    limit=None,
    img_id=None,
    use_half=False,
    use_half_decoder=False,
    use_compile=False,
    use_compile_max_autotune=False,
    use_compile_decoder=False,
    use_quantize=False,
    num_workers=0,
    use_cudagraph_trees=True
):
    if not use_cudagraph_trees:
        from torch._inductor import config as tritonconfig
        tritonconfig.triton.cudagraph_trees = False

    # https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
    # largest to smallest: vit_h, vit_l, vit_b
    model_type_to_checkpoint = {
        'vit_h': f'{sam_checkpoint_base_path}/sam_vit_h_4b8939.pth',
        'vit_l': f'{sam_checkpoint_base_path}/sam_vit_l_0b3195.pth',
        'vit_b': f'{sam_checkpoint_base_path}/sam_vit_b_01ec64.pth',
    }

    checkpoint_path = model_type_to_checkpoint[sam_model_type]

    # load SAM
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path).cuda()

    predictor = SamPredictor(sam)

    predictor.model.image_encoder = predictor.model.image_encoder.eval()
    if use_half:
        predictor.model.image_encoder = predictor.model.image_encoder.half()

    predictor.model.prompt_encoder = predictor.model.prompt_encoder.eval()
    predictor.model.mask_decoder = predictor.model.mask_decoder.eval()
    if use_half_decoder:
        predictor.model.prompt_encoder = predictor.model.prompt_encoder.half()
        predictor.model.mask_decoder = predictor.model.mask_decoder.half()

    if use_quantize:
        assert use_half
        # assert use_compile_max_autotune
        assert not use_compile_decoder
        apply_dynamic_quant(predictor.model.image_encoder)
        from torch._inductor import config as tritonconfig
        tritonconfig.triton.unique_kernel_names = True
        # tritonconfig.aggressive_fusion = True
        # tritonconfig.triton.tiling_prevents_pointwise_fusion = True
        tritonconfig.epilogue_fusion_first = True

    if use_compile:
        assert not use_compile_max_autotune
        predictor.model.image_encoder = torch.compile(
            predictor.model.image_encoder)

    if use_compile_max_autotune:
        assert not use_compile
        predictor.model.image_encoder = torch.compile(
            predictor.model.image_encoder, mode="max-autotune-no-cudagraphs")
        # predictor.model.image_encoder = torch.compile(predictor.model.image_encoder, mode="max-autotune")

    # limit = batch_size # 20
    # metrics = run_do_eval_with_profile(

    compile_create_top_score_ious = use_compile_decoder
    silent = True

    cache = diskcache.Cache(point_sampling_cache_dir)
    # make sure you clear the cache if you change the point sampling algorithm
    # cache.clear()

    coco_img_ids, cat_id_to_cat, catIds, coco = setup_coco_img_ids(
        coco_root_dir, coco_slice_name, coco_category_names, img_id)

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
    results, batch_ms = build_results(batched_data_iter,
                                      predictor,
                                      mask_debug_out_dir,
                                      batch_size,
                                      compile_create_top_score_ious)

    results = [[r[0], r[1], r[2], r[3].item()] for r in results]

    batch_ms_p50 = batch_ms.quantile(0.5, interpolation='nearest').item()
    ms_per_img = (batch_ms_p50 / batch_size)
    img_s = 1000 / ms_per_img

    mIoU = calculate_miou(results, mask_debug_out_dir, silent, cat_id_to_cat)
    max_memory_allocated = torch.cuda.max_memory_allocated()

    if print_header:
        print(",".join(["sam_model_type", "batch_size", "max_memory_allocated", "img_s", "mIoU", "use_compile",
              "use_half", "use_quantize", "use_half_decoder", "use_compile_decoder", "use_cudagraph_trees", "num_workers"]))
    print(",".join(map(str, [sam_model_type, batch_size, max_memory_allocated, img_s, mIoU, use_compile,
          use_half, use_quantize, use_half_decoder, use_compile_decoder, use_cudagraph_trees, num_workers])))


if __name__ == '__main__':
    fire.Fire(run)
