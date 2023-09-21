import tqdm
import torch
import fire
from metrics import calculate_miou, create_result_entry
from data import build_data, setup_coco_img_ids
import math

torch._dynamo.config.cache_size_limit = 50000


def unbind_jagged(device, data, sizes, offsets):
    if data is None:
        return None
    data = data.to(device=device, non_blocking=True)
    return [data[offsets[batch_idx]:offsets[batch_idx+1]].view(sizes[batch_idx]) for batch_idx in range(len(sizes))]

PADDED_TENSOR=None

# Preallocate a "landing" Tensor for incoming data and reuse it across launches.
def pad_to_batch_size(batch, batch_size, device):
    assert batch.dim() == 4
    assert batch.is_pinned()
    global PADDED_TENSOR
    if PADDED_TENSOR is None:
        batch = batch.to(device=device, non_blocking=True)
        full_batch_size = (batch_size, batch.size(1), batch.size(2), batch.size(3))
        first_entry = batch[0].unsqueeze(0)
        repeat_first_entry = first_entry.expand(full_batch_size)
        padded_batch = torch.cat([batch, repeat_first_entry[batch.size(0):batch_size]], dim=0)
        assert padded_batch.size() == full_batch_size
        PADDED_TENSOR = padded_batch
    PADDED_TENSOR[:batch.size(0)].copy_(batch, non_blocking=True)
    return PADDED_TENSOR

def get_features_batch(encoder, input_image_batch, pad_input_image_batch, batch_size, device):
    if pad_input_image_batch:
        features_batch = encoder(pad_to_batch_size(input_image_batch, batch_size, device))
        return features_batch[:input_image_batch.size(0)]
    return encoder(input_image_batch)

def build_results_batch_nested(predictor, batch, batch_size, pad_input_image_batch):
    encoder = predictor.model.image_encoder
    device = predictor.device

    input_image_batch = batch[0]
    # The number of valid data points varies slightly per batch
    orig_input_image_batch_size = input_image_batch.size(0)
    if input_image_batch is None:
        return (None, None, None)
    if batch[1] is None:
        return (None, None, None)

    with torch.autograd.profiler.record_function("nt data transfer"):
        datapoints = list(zip(*(batch[7:])))
        nt_coords = batch[1].to(device=device, non_blocking=True)
        nt_gt_masks = batch[4].to(device=device, non_blocking=True)
        nt_fg_labels = torch.nested.nested_tensor([torch.ones(
            (coords.size(0), 1), dtype=torch.int, device=device) for coords in nt_coords.unbind()])
        if pad_input_image_batch:
            # Pad to a static shape to avoid recompilation
            input_image_batch = pad_to_batch_size(input_image_batch, batch_size, device)
        else:
            input_image_batch = input_image_batch.to(device=device, non_blocking=True)

    # We explicitly exclude data transfers from the timing to focus
    # only on the kernel performance.
    # Next we synchronize and set two events to start timing.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.autograd.profiler.record_function("timed region"):
        with torch.autograd.profiler.record_function("image encoder"):
            features_batch = encoder(input_image_batch)
            features_batch = features_batch[:orig_input_image_batch_size]

        with torch.autograd.profiler.record_function("nt predict_torch"):
            predictor.reset_image()
            predictor.original_sizes = [d[1].shape[:2] for d in datapoints]
            predictor.input_sizes = [d[2] for d in datapoints]
            predictor.features_batch = features_batch
            predictor.is_image_set = True
            nt_coords = nt_coords.unsqueeze(2)
            masks, scores, logits = predictor.predict_torch(
                point_coords=nt_coords,
                point_labels=nt_fg_labels,
                multimask_output=True,
            )
            result_batch = [create_result_entry(d[0], g, m, s, d[3]) for (m, s, d, g) in zip(masks.unbind(),
                                                                                     scores.unbind(), datapoints,
                                                                                     nt_gt_masks.unbind())]
        # After all kernels have been launched we synchronize again and measure
        # the amount of time spent on the GPU. This is a fairly tight measurement
        # around the launched GPU kernels and excludes data movement from host
        # to device.
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
    return sum(result_batch, []), orig_input_image_batch_size, elapsed_time

def build_results_batch(predictor, batch, batch_size, pad_input_image_batch):
    encoder = predictor.model.image_encoder
    device = predictor.device

    input_image_batch = batch[0]
    # The number of valid data points varies slightly per batch
    orig_input_image_batch_size = input_image_batch.size(0)
    if input_image_batch is None:
        return (None, None, None)

    with torch.autograd.profiler.record_function("data transfer"):
        coords_lists = unbind_jagged(*([device] + batch[1:4]))
        gt_masks_lists = unbind_jagged(*([device] + batch[4:7]))
        if coords_lists is None:
            return (None, None, None)
        datapoints = list(zip(*(batch[7:] + [coords_lists, gt_masks_lists])))
        if pad_input_image_batch:
            # Pad to a static shape to avoid recompilation
            input_image_batch = pad_to_batch_size(input_image_batch, batch_size, device)
        else:
            input_image_batch = input_image_batch.to(device=device, non_blocking=True)

    # We explicitly exclude data transfers from the timing to focus
    # only on the kernel performance.
    # Next we synchronize and set two events to start timing.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.autograd.profiler.record_function("timed region"):
        with torch.autograd.profiler.record_function("image encoder"):
            features_batch = encoder(input_image_batch)
            features_batch = features_batch[:orig_input_image_batch_size]

        with torch.autograd.profiler.record_function("predict_torch"):
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
                entry = create_result_entry(anns, gt_masks, masks, scores, idx)
                result_batch += entry

        # After all kernels have been launched we synchronize again and measure
        # the amount of time spent on the GPU. This is a fairly tight measurement
        # around the launched GPU kernels and excludes data movement from host
        # to device.
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
    return result_batch, orig_input_image_batch_size, elapsed_time


def build_results(batched_data_iter,
                  predictor,
                  mask_debug_out_dir,
                  batch_size,
                  use_compile,
                  use_compile_decoder,
                  use_nested_tensor,
                  pad_input_image_batch):

    # TODO: Re-enable this for datapoints
    assert not use_compile_decoder

    batch_runner = None
    if use_nested_tensor:
        batch_runner = build_results_batch_nested
    else:
        batch_runner = build_results_batch

    results = []
    batch_idx = 0
    num_images = 0
    num_batches = 0
    elapsed_time = 0
    partial_batch = False
    for batch in tqdm.tqdm(batched_data_iter):
        with torch.no_grad():
            if batch_idx == 0:
                with torch.autograd.profiler.record_function("compilation and warmup"):
                    if str(use_compile) != "False":
                        predictor.model.image_encoder = torch.compile(predictor.model.image_encoder, mode=use_compile)
                    # Run first batch a few times for warmup and exclude it from the final timings
                    for _ in range(3):
                        _ = batch_runner(predictor, batch, batch_size, pad_input_image_batch)
            result_batch, num_datapoints, kernel_time = batch_runner(predictor, batch, batch_size, pad_input_image_batch)
            if result_batch is not None:
                results += result_batch
        # We expect a partial batch to only happens once at the end
        assert not partial_batch
        # Only measure timing on full batches
        if num_datapoints == batch_size:
            num_images += num_datapoints
            num_batches += 1
            # We consistently exclude the last (512 - filtered) images
            # Since batch sizes must be powers of two and less than
            # or equal 512 this ensures consistent timing across varying
            # batch sizes.
            if num_images <= 4488:
                elapsed_time += kernel_time
        else:
            partial_batch = True
        batch_idx += 1

    avg_ms_per_img = None
    if num_images > 0:
        avg_ms_per_img = elapsed_time
        avg_ms_per_img = avg_ms_per_img / num_images

    return results, avg_ms_per_img, num_batches, num_images


def identity_runner(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def profile_top_runner(fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    return result


def memory_runner(path, fn, *args, **kwargs):
    print("Start memory recording")
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100000, trace_alloc_record_context=True)
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    print("Finish memory recording")
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(snapshot, f)
    # Use to convert pickle file into html
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html
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
    use_compile="False",
    use_compile_decoder=False,
    compress=None,
    epilogue_fusion_first=False,
    num_workers=0,
    use_nested_tensor=False,
    use_rel_pos=True,
    pad_input_image_batch=True,
    profile_path=None,
    profile_top=False,
    memory_path=None,
    use_local_sam_fork=False,
):
    from torch._inductor import config as tritonconfig
    tritonconfig.triton.unique_kernel_names = True
    tritonconfig.epilogue_fusion_first = epilogue_fusion_first

    # Batch size needs to be a multiple of two and at most 512.
    assert math.log2(batch_size).is_integer()
    assert batch_size <= 512

    # https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
    # largest to smallest: vit_h, vit_l, vit_b
    model_type_to_checkpoint = {
        'vit_h': f'{sam_checkpoint_base_path}/sam_vit_h_4b8939.pth',
        'vit_l': f'{sam_checkpoint_base_path}/sam_vit_l_0b3195.pth',
        'vit_b': f'{sam_checkpoint_base_path}/sam_vit_b_01ec64.pth',
    }

    if use_local_sam_fork:
        from local_segment_anything_fork import sam_model_registry, SamPredictor
    else:
        from segment_anything import sam_model_registry, SamPredictor
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

    for block in predictor.model.image_encoder.blocks:
        block.attn.use_rel_pos = use_rel_pos

    if compress == "dynamic_quant":
        from dynamic_quant import apply_dynamic_quant
        apply_dynamic_quant(predictor.model.image_encoder)
    elif compress == "static_quant":
        from static_quant import apply_static_quant
        apply_static_quant(predictor.model.image_encoder)
        from pathlib import Path
        weights_path = Path(f"{sam_model_type}_{batch_size}_static_quant_weights.ptk")
        if weights_path.exists() and weights_path.is_file():
            print("Loading static quantization weights")
            weights = torch.load(f"{sam_model_type}_{batch_size}_static_quant_weights.ptk")
            from static_quant import set_x_absmax
            set_x_absmax(predictor.model.image_encoder, weights)
    elif compress == "sparse":
        from sparse import apply_sparse
        apply_sparse(predictor.model.image_encoder)
    elif compress == "int4_dynamic_quant_sparse":
        from dynamic_quant_sparse import apply_int4_dynamic_quant_sparse
        apply_int4_dynamic_quant_sparse(predictor.model.image_encoder)
    elif compress == "static_quant_sparse":
        raise NotImplementedError(f"Unsupported compress {compress}")
    elif compress == "sparse":
        raise NotImplementedError(f"Unsupported compress {compress}")
    else:
        assert compress is None, f"Unsupported compress mode {compress}"


    coco_img_ids_, cat_id_to_cat, catIds, coco = setup_coco_img_ids(
        coco_root_dir, coco_slice_name, coco_category_names, img_id)

    coco_img_ids = []
    for imgId in coco_img_ids_:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(anns) != 0:
            coco_img_ids.append(imgId)

    build_batch = build_data(coco_img_ids,
                             coco,
                             catIds,
                             coco_root_dir,
                             coco_slice_name,
                             point_sampling_cache_dir,
                             predictor,
                             use_half,
                             use_half_decoder,
                             use_nested_tensor,
                             pad_input_image_batch)

    limit = len(coco_img_ids) if limit is None else limit
    batched_data_iter = torch.utils.data.DataLoader(list(range(limit)),
                                                    batch_size=batch_size,
                                                    collate_fn=build_batch,
                                                    num_workers=num_workers,
                                                    pin_memory=True)
    runner = identity_runner

    if profile_path is not None:
        import functools
        runner = functools.partial(profiler_runner, profile_path)

    if profile_top:
        runner = profile_top_runner

    if memory_path is not None:
        import functools
        runner = functools.partial(memory_runner, memory_path)

    results, avg_ms_per_img, num_batches, num_images = runner(build_results,
                                                              batched_data_iter,
                                                              predictor,
                                                              mask_debug_out_dir,
                                                              batch_size,
                                                              use_compile,
                                                              use_compile_decoder,
                                                              use_nested_tensor,
                                                              pad_input_image_batch)

    if compress == "static_quant":
        from static_quant import get_x_absmax
        weights = get_x_absmax(predictor.model.image_encoder)
        print("Saving static quantization weights")
        torch.save(weights, f"{sam_model_type}_{batch_size}_static_quant_weights.ptk")

    results = [[r[0], r[1], r[2], r[3].item()] for r in results]

    img_s, batch_ms_batch_size = None, None
    if avg_ms_per_img is not None:
        img_s = 1000 / avg_ms_per_img
        batch_ms_batch_size = (avg_ms_per_img * num_images) / num_batches / batch_size

    mIoU = calculate_miou(results, mask_debug_out_dir, True, cat_id_to_cat)
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20

    if print_header:
        print(",".join(["sam_model_type", "batch_size", "memory(MiB)", "memory(%)", "img_s(avg)", "batch_ms(avg)/batch_size", "mIoU", "use_compile",
              "use_half", "compress", "epilogue_fusion_first", "use_half_decoder", "use_compile_decoder", "use_nested_tensor", "use_rel_pos", "pad_input_image_batch", "num_workers", "num_batches", "num_images", "profile_path", "memory_path"]))
    print(",".join(map(str, [sam_model_type, batch_size, max_memory_allocated_bytes, max_memory_allocated_percentage, img_s, batch_ms_batch_size, mIoU, use_compile,
          use_half, compress, epilogue_fusion_first, use_half_decoder, use_compile_decoder, use_nested_tensor, use_rel_pos, pad_input_image_batch, num_workers, num_batches, num_images, profile_path, memory_path])))


if __name__ == '__main__':
    fire.Fire(run)
