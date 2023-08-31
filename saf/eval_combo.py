"""
Runs evaluation of SAM on the validation set of COCO Val 2017

COCO data setup:
1. go here: https://cocodataset.org/#download
2. download 2017 Val images
3. download 2017 Train/Val annotations
4. unzip everything and put in {coco_root_dir}

SAM checkpoint setup:
1. go here: https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints
2. download all the checkpoints to a directory
3. pass in that directory via {sam_checkpoint_base_path}
"""

# this was adapted from https://www.internalfb.com/intern/anp/view/?kernel=default&id=3976727


# for now, speed up point calculation with diskcache
# depending on requirements of our eval, this may have to change in the future
import fire

import torch
from segment_anything import sam_model_registry, SamPredictor

from eval import do_eval

torch._dynamo.config.cache_size_limit = 5000


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
        import sys
        import os
        # smoothquant_path = '/'.join(os.getcwd().split('/')[:-1] + \
        #     ['ao_benchmarks/experimental/smoothquant'])
        # sys.path.insert(1, smoothquant_path)
        from quant import apply_dynamic_quant
        apply_dynamic_quant(predictor.model.image_encoder)
        from torch._inductor import config as tritonconfig
        tritonconfig.triton.unique_kernel_names = True
        # tritonconfig.aggressive_fusion = True
        # tritonconfig.triton.tiling_prevents_pointwise_fusion = True
        tritonconfig.epilogue_fusion_first = True

    if use_compile:
        assert not use_compile_max_autotune
        predictor.model.image_encoder = torch.compile(predictor.model.image_encoder)

    if use_compile_max_autotune:
        assert not use_compile
        predictor.model.image_encoder = torch.compile(predictor.model.image_encoder, mode="max-autotune-no-cudagraphs")
        # predictor.model.image_encoder = torch.compile(predictor.model.image_encoder, mode="max-autotune")

    # limit = batch_size # 20
    # metrics = run_do_eval_with_profile(
    metrics = do_eval(
        predictor, coco_root_dir, coco_slice_name, coco_category_names,
        point_sampling_cache_dir, mask_debug_out_dir, limit, img_id,
        batch_size=batch_size, report_batch_timings=True, silent=True,
        num_workers=num_workers, use_half=use_half, save_inference_masks=False,
        compile_create_top_score_ious=use_compile_decoder, use_half_decoder=use_half_decoder)
    batch_ms = metrics['batch_ms']
    results_df = metrics['results_df']

    batch_ms_p50 = batch_ms.quantile(0.5, interpolation='nearest').item()
    ms_per_img = (batch_ms_p50 / batch_size)
    img_s = 1000 / ms_per_img
    mIoU = results_df['iou'].agg(['mean', 'count'])['mean']
    max_memory_allocated = torch.cuda.max_memory_allocated()
    if print_header:
        print(",".join(["sam_model_type", "batch_size", "max_memory_allocated", "img_s", "mIoU", "use_compile", "use_half", "use_quantize", "use_half_decoder", "use_compile_decoder", "use_cudagraph_trees", "num_workers"]))
    print(",".join(map(str, [sam_model_type, batch_size, max_memory_allocated, img_s, mIoU, use_compile, use_half, use_quantize, use_half_decoder, use_compile_decoder, use_cudagraph_trees, num_workers])))
    # from torch._inductor import config
    # print(
    #         "torch._inductor.config.epilogue_fusion: ",
    #         torch._inductor.config.epilogue_fusion)


if __name__ == '__main__':
    fire.Fire(run)
