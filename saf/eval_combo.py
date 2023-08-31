import tqdm
import torch
import fire
from metrics import calculate_miou, create_result_entry
from data import build_data, setup_coco_img_ids
from quant import apply_dynamic_quant
from segment_anything import sam_model_registry, SamPredictor

torch._dynamo.config.cache_size_limit = 5000


def unbind_jagged(device, data, sizes, offsets):
    if data is None:
        return None
    data = data.to(device=device, non_blocking=True)
    return [data[offsets[batch_idx]:offsets[batch_idx+1]].view(sizes[batch_idx]) for batch_idx in range(len(sizes))]


def build_results_batch(predictor, batch):
    encoder = predictor.model.image_encoder
    device = predictor.device

    input_image_batch = batch[0]
    if input_image_batch is None:
        return None
    input_image_batch = input_image_batch.to(device=device, non_blocking=True)
    coords_lists = unbind_jagged(*([device] + batch[1:4]))
    gt_masks_lists = unbind_jagged(*([device] + batch[4:7]))
    if coords_lists is None:
        return None
    features_batch = encoder(input_image_batch)
    datapoints = zip(*(batch[7:] + [coords_lists, gt_masks_lists]))

    result_batch = []
    for batch_idx, (anns, image, input_size, idx, coords, gt_masks) in enumerate(datapoints):
        features = features_batch.narrow(0, batch_idx, 1)
        predictor.reset_image()
        predictor.original_size = image.shape[:2]
        predictor.input_size = input_size
        predictor.features = features
        predictor.is_image_set = True
        coords = coords.unsqueeze(1)
        fg_labels = torch.ones(
            (coords.size(0), 1), dtype=torch.int, device=device)
        # TODO: Break this up further to batch more computation.
        masks, scores, logits = predictor.predict_torch(
            point_coords=coords,
            point_labels=fg_labels,
            multimask_output=True,
        )
        result_batch += create_result_entry(anns, gt_masks, masks, scores, idx)
    return result_batch


def build_results(batched_data_iter,
                  predictor,
                  mask_debug_out_dir,
                  batch_size,
                  use_compile_decoder):

    # TODO: Re-enable this for datapoints
    assert not use_compile_decoder

    results = []
    batch_ms = []
    for batch in tqdm.tqdm(batched_data_iter):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():
            result_batch = build_results_batch(predictor, batch)
            if result_batch is not None:
                results += result_batch

        end_event.record()
        torch.cuda.synchronize()
        batch_ms.append(start_event.elapsed_time(end_event))

    return results, torch.tensor(batch_ms)


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
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint_path).cuda()
    predictor = SamPredictor(sam)

    def prep_model(model, use_half):
        if use_half:
            return model.eval().half()
        return model.eval()

    predictor.model.image_encoder = prep_model(
        predictor.model.image_encoder, use_half)
    predictor.model.prompt_encoder = prep_model(
        predictor.model.prompt_encoder, use_half_decoder)
    predictor.model.mask_decoder = prep_model(
        predictor.model.mask_decoder, use_half_decoder)

    if use_quantize:
        apply_dynamic_quant(predictor.model.image_encoder)
        from torch._inductor import config as tritonconfig
        tritonconfig.triton.unique_kernel_names = True
        tritonconfig.epilogue_fusion_first = True

    if use_compile:
        assert not use_compile_max_autotune
        predictor.model.image_encoder = torch.compile(
            predictor.model.image_encoder)

    if use_compile_max_autotune:
        assert not use_compile
        predictor.model.image_encoder = torch.compile(
            predictor.model.image_encoder, mode="max-autotune-no-cudagraphs")

    coco_img_ids, cat_id_to_cat, catIds, coco = setup_coco_img_ids(
        coco_root_dir, coco_slice_name, coco_category_names, img_id)

    build_batch = build_data(coco_img_ids,
                             coco,
                             catIds,
                             coco_root_dir,
                             coco_slice_name,
                             point_sampling_cache_dir,
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
                                      use_compile_decoder)

    results = [[r[0], r[1], r[2], r[3].item()] for r in results]

    batch_ms_p50 = batch_ms.quantile(0.5, interpolation='nearest').item()
    ms_per_img = (batch_ms_p50 / batch_size)
    img_s = 1000 / ms_per_img

    mIoU = calculate_miou(results, mask_debug_out_dir, True, cat_id_to_cat)
    max_memory_allocated = torch.cuda.max_memory_allocated()

    if print_header:
        print(",".join(["sam_model_type", "batch_size", "max_memory_allocated", "img_s", "mIoU", "use_compile",
              "use_half", "use_quantize", "use_half_decoder", "use_compile_decoder", "use_cudagraph_trees", "num_workers"]))
    print(",".join(map(str, [sam_model_type, batch_size, max_memory_allocated, img_s, mIoU, use_compile,
          use_half, use_quantize, use_half_decoder, use_compile_decoder, use_cudagraph_trees, num_workers])))


if __name__ == '__main__':
    fire.Fire(run)
