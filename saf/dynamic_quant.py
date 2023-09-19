import torch
import torch
import torch.nn as nn
from typing import Tuple, Optional

import torch
from torch._dynamo import is_compiling as dynamo_is_compiling
from int_mm import _int_mm_dequant

def quantize_activation_per_token(t, scales):
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t

def quantize_activation_per_token_absmax(t):
    n_bits = 8
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
    # want float scales to avoid overflows
    scales = t.abs().max(dim=-1, keepdim=True)[0].float()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales

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
        def quant_int8_dynamic_per_token_linear(
            x_vals_int8,
            x_scales,
            w_vals_int8_t,
            w_scales,
            bias,
            out_dtype=torch.float32,
        ):

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


                # TODO(before land): add test case for input with bsz
                tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
                x_scales_flat = x_scales.view(tmp.size(0), 1)
                w_scales_flat = w_scales.unsqueeze(0)
                # y_dot_int32 = torch._int_mm(tmp, w_vals_int8_t)

                #
                # 2. rescale the output
                #
                # in cases with large matrices, y_dot_int32 can grow sufficiently
                # large that y_dot_int32 * a float16 scale is greater than the maximum
                # value of a float 16, (which results in a value of inf even if multiplying
                # by the other scale would bring it within the expected range)

                assert x_scales.dtype == torch.float, f"x_scales needs to be a torch.float32 but got {x_scales.dtype}"

                # y = y_dot_int32 * x_scales_flat * w_scales_flat
                # # can downcast only at the very end
                # y = y.to(out_dtype)

                # y = _int_mm_dequant(tmp, w_vals_int8_t, x_scales_flat, w_scales_flat, out_dtype)
                y = torch.ops.my_int_mm.int_mm(tmp, w_vals_int8_t, x_scales_flat, w_scales_flat, out_dtype)
                return y.reshape(*x_vals_int8.shape[:-1], -1)

            # like F.linear, but with int8 dynamic quantization of activation,
            # and a quantized weight
            mm_out = quant_int8_per_token_matmul(
                x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype)
            if bias is not None:
                mm_out += bias
            return mm_out

        x_vals_int8, x_scales = quantize_activation_per_token_absmax(X)
        Y = quant_int8_dynamic_per_token_linear(x_vals_int8, x_scales, self.W_int_repr_t, self.W_scales, self.bias, X.dtype)
        return Y

    @classmethod
    def from_float(cls, mod: torch.nn.Linear) -> 'DynamicallyPerAxisQuantizedLinear':
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
            zero_point = torch.zeros(
                min_val_neg.size(), dtype=torch.int64, device=device)

            # quantize based on qmin/qmax/scale/zp
            # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
            x_div = x.transpose(0, 1) / scale
            x_round = torch.round(x_div)
            x_zp = x_round + zero_point
            x_zp = x_zp.transpose(0, 1)
            quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

            return quant, scale, zero_point
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

def apply_dynamic_quant(model):
    from utils import replace_with_custom_fn_if_matches_filter
    replace_with_custom_fn_if_matches_filter(
        model,
        DynamicallyPerAxisQuantizedLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))
