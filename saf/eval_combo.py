import torch
import torch.nn as nn
from typing import Tuple, Optional

import torch
from torch._dynamo import is_compiling as dynamo_is_compiling

def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    r"""
    This function wraps torch._int_mm and avoids several undesirable behaviors of the function for certain inputs while still
    returning correct results and being torch.compiled in a performant way.

    Assumes both tensors have dimension of 2.

    Note: no error checking for torch.compiled path, if input.shape = [i, j] and j<=16 then the triton kernel
    will error.

    Args:
        input (Tensor, int8): the first tensor to be multiplied
        mat2 (Tensor, int8): the second tensor to be multiplied

    Return:
        out (Tensor, int32): the result of the matmul with device matching that of the inputs
    """

    # torch.compile path
    if dynamo_is_compiling():
        return torch._int_mm(input, mat2)

    # error checking for cublas path
    assert (
        mat2.device == input.device
    ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
    device_cpu = "cpu" in [mat2.device.type, input.device.type]
    # with input.shape = [i,j] and mat2.shape = [j,k]
    i_is_strictly_greater_than_16 = input.shape[0] > 16
    j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
    k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
    bad_dimensions_for_cublas = not (
        i_is_strictly_greater_than_16
        and j_is_nonzero_multiple_of_8
        and k_is_nonzero_multiple_of_8
    )

    if device_cpu or bad_dimensions_for_cublas:
        # fallback path
        return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
            input.device.type
        )

    # cublas paths
    if not mat2.is_contiguous():  # silently gives incorrect result without this
        mat2 = mat2.contiguous()
    if (not input.is_contiguous()) and (
        input.shape[0] % 8 != 0
    ):  # gives cryptic error without this
        input = (
            input.contiguous()
        )  # (it seems the transpose makes cublas check the above j constraint on i)
    return torch._int_mm(input, mat2)


# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736
def dynamically_quantize_per_tensor(
    x,
    quant_min,
    quant_max,
    target_dtype,
    qscheme=torch.per_tensor_affine,  # for now, reuse existing qscheme enum
):
    # assumes affine quantization

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    if qscheme == torch.per_tensor_affine:

        # get min and max
        # TODO(future): make torch.aminmax work on cpu-half
        # min_val, max_val = torch.aminmax(x)
        min_val = torch.min(x)
        max_val = torch.max(x)

        # calculate scale and zero point based on min and max
        # reference: https://fburl.com/code/srbiybme
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # TODO(future): make torch.clamp with scalar work on cpu-half
        scale = torch.clamp(scale, min=eps).reshape(1)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # quantize based on qmin/qmax/scale/zp
        # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
        quant = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max).to(target_dtype)

    else:
        assert qscheme == torch.per_tensor_symmetric, f"unsupported qscheme {qscheme}"
        # assert quant_min == -1 * quant_max, "unsupported quant_min/quant_max"
        amax = torch.max(torch.abs(x))
        scale = amax / (float(quant_max - quant_min) / 2)
        scale = torch.clamp(scale, min=eps).reshape(1)
        quant = torch.clamp(torch.round(x / scale), quant_min, quant_max).to(target_dtype)
        # do not create a tensor for zero_point as this is expensive
        zero_point = None

    return quant, scale, zero_point

# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified
def quantize_activation_per_token_absmax(t):
    n_bits = 8
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
    scales = t.abs().max(dim=-1, keepdim=True)[0].float() # want float scales to avoid overflows
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales

def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point

# reference: https://fburl.com/code/vfsygwd0
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    y = int_repr.to(out_dtype)
    if zero_point is not None:
        y -= zero_point
    return y * scale

# reference: https://fburl.com/code/org0fmi3
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    # assumes axis is 0
    y = int_repr.transpose(0, 1)
    y = y.to(out_dtype)
    y = y - zero_points
    y = y * scales
    y = y.transpose(0, 1)
    return y

def quant_int8_dynamic_linear(
    x,
    x_quant_min,
    x_quant_max,
    x_q_dtype,
    w_vals_int8_t,
    w_scales,
    w_vals_int8_t_sums_int64,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(
        x, x_quant_min, x_quant_max, x_q_dtype)
    # w_vals_int8_t_sums_int64 = w_vals_int8_t.sum(dim=0)
    mm_out = quant_int8_matmul(
        x_vals_int8, x_scale, x_zp, w_vals_int8_t, w_vals_int8_t_sums_int64,
        w_scales, out_dtype)
    if bias is not None:
        mm_out += bias
    return mm_out

def quant_int8_matmul(
    x_vals_int8,
    x_scale,
    x_zp,
    w_vals_int8_t,
    w_vals_int8_t_sums_int64,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # correctness, and things like aligning accumulation behaviors and
    # performance optimizations are left for a future PR.
    # Assumes that weight quantization is symmetric, i.e. w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw (dot(X_i, W_j) - zx * sum(W_j))
    #

    assert x_vals_int8.dtype in (torch.uint8, torch.int8), \
        f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8_t.dtype == torch.int8, \
        f'w dtype {w_vals_int8_t.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, \
        f'{w_scales.dtype} does not match {out_dtype}'

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    print("1ASDF")
    # TODO(before land): add test case for input with bsz
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)
    y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)

    # TODO(future): consider using integer arithmetic throughout, although
    # TBD if that is actually faster on GPUs
    # need to use 32 bits here to prevent overflow for large shapes,
    # 16 bits is not enough
    y_dot_float32 = y_dot_int32.to(torch.float16) #.to(torch.float32)

    #
    # 2. connect it all together
    #

    # mm_unscaled has to stay in float32 for the next two lines to prevent overflow
    mm_unscaled_float32 = (y_dot_float32 - (x_zp * w_vals_int8_t_sums_int64))
    y = x_scale * w_scales * mm_unscaled_float32
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y

def quant_int8_dynamic_per_token_linear(
    x,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = quant_int8_per_token_matmul(
        x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype)
    if bias is not None:
        mm_out += bias
    return mm_out

def quant_int8_per_token_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8_t,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # Assumes that activation and weight quantization are symmetric,
    # i.e. act_zp and w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    #

    assert x_vals_int8.dtype == torch.int8, \
        f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8_t.dtype == torch.int8, \
        f'w dtype {w_vals_int8_t.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, \
        f'{w_scales.dtype} does not match {out_dtype}'

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    # print("0ASDF")
    # TODO(before land): add test case for input with bsz
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)
    y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)
    y_dot_float64 = y_dot_int32.to(torch.float64)

    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    assert x_scales.dtype == torch.float, f"x_scales needs to be a torch.float32 but got {x_scales.dtype}"

    # print("x_scales: ", x_scales.dtype)
    # print("w_scales: ", w_scales.dtype)
    y = y_dot_float64 * x_scales * w_scales
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y

class DynamicallyPerAxisQuantizedLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`, implementing dynamic quantization on
    the input across all axes except for the last axis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer.

        This method applies dynamic quantization to the input tensor across all axes except
        the last axis using the `quant_int8_dynamic_per_token_linear` function.

        Args:
            X (torch.Tensor): The input tensor to the quantized linear layer.

        Returns:
            torch.Tensor: The output tensor after the quantized matmul and rescale.

        """
        # import pdb; pdb.set_trace()
        Y = quant_int8_dynamic_per_token_linear(
            X, self.W_int_repr_t, self.W_scales, self.bias, X.dtype)
        return Y

    @classmethod
    def from_float(cls, mod: torch.nn.Linear) -> 'DynamicallyPerAxisQuantizedLinear':
        """
        Converts a `mod` of class `torch.nn.Linear` to the dynamically quantized version of it.

        Note: this class does not require calibration.

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            DynamicallyPerAxisQuantizedLinear: The converted quantized linear module.

        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias=mod.bias is not None)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        W_int_repr, W_scales, _W_zps = dynamically_quantize_per_channel(
            mod.weight, -128, 127, torch.int8)
        new_mod.register_buffer('W_int_repr_t', W_int_repr.contiguous().t())
        new_mod.W_scales = nn.Parameter(W_scales)
        new_mod.bias = mod.bias
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

"""
Quantization API stuff which is not specific to SmoothQuant

Note: this is throwaway code for fast results on Blueberry, this is not
intended to be the actual long term quantization API for server GPUs.
"""

import torch


def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=''
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == '':
            new_fqn = name
        else:
            new_fqn = f'{cur_fqn}.{name}'
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn)

def apply_dynamic_quant(model):
    replace_with_custom_fn_if_matches_filter(
        model,
        DynamicallyPerAxisQuantizedLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))

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
