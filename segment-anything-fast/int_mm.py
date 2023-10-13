import torch

import triton
import triton.language as tl

@triton.jit
def matmul_kernel_with_block_pointers(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, s1_ptr, s2_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_s1m, stride_s1n,
        stride_s2m, stride_s2n,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr) #, boundary_check=(0, 1))
        b = tl.load(b_block_ptr) #, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator #.to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    s1_ptrs = s1_ptr + offs_m[:, None] * stride_s1m + offs_n[None, :] * stride_s1n
    s2_ptrs = s2_ptr + offs_m[:, None] * stride_s2m + offs_n[None, :] * stride_s2n
    s1 = tl.load(s1_ptrs)
    c = c * s1
    s2 = tl.load(s2_ptrs)
    c = c * s2
    c = c.to(tl.bfloat16)
    # Epilogue
    tl.store(c_block_ptr, c) #, boundary_check=(0, 1))

import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, number, *args, **kwargs):
    try:
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
    except:
        return None
    return t0.timeit(number=number).mean * 1e6

def _autotune(configs, function):
    best = None
    best_config = None
    for i, config in enumerate(configs):
        t_config = benchmark_torch_function_in_microseconds(function, 1, *config)
        if t_config is not None:
            if best is not None:
                if t_config < best * 2:
                    t_config = benchmark_torch_function_in_microseconds(function, 10, *config)
                if t_config < best:
                    best = t_config
                    best_config = config
            else:
                t_config = benchmark_torch_function_in_microseconds(function, 10, *config)
                best = t_config
                best_config = config
        print(f"\ri: {i+1}/{len(configs)} ", str(config), " :", str(t_config), "\t\t", end='')
    print("")
    return best, best_config

def _load_best_configs():
    from pathlib import Path
    saved_configs = Path("int_mm_configs_a100.p")
    if saved_configs.is_file():
        import pickle
        with open(saved_configs, 'rb') as f:
            print(f"Loading best configs from file {saved_configs}")
            return pickle.load(f)

def _save_best_configs(best_configs):
    from pathlib import Path
    saved_configs = Path("int_mm_configs_a100.p")
    with open(saved_configs, 'wb') as f:
        import pickle
        print(f"Saving best configs to file {saved_configs}")
        pickle.dump(best_configs, f)

def _create_best_configs_key(key_tensors):
    key = sum([[k.size(), k.stride()] for k in key_tensors], [])
    key = tuple(key)
    return key

# Built on an A100 80GB
BEST_CONFIGS = None

def _find_config(key_tensors, function):
    global BEST_CONFIGS
    if BEST_CONFIGS is None:
        BEST_CONFIGS = _load_best_configs()
    if BEST_CONFIGS is None:
        BEST_CONFIGS = {}
    key = _create_best_configs_key(key_tensors)
    if key in BEST_CONFIGS:
        return BEST_CONFIGS[key], False

    print(f"Could not find a config for key {key}")
    import itertools
    # (BLOCK_M, BLOCK_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps)
    configs = itertools.product([32, 64, 128, 256], [32, 64, 128, 256], [32, 64], [4, 8], [3, 4, 5], [2, 4, 8])
    configs = list(filter(lambda x: not(x[0] == 256 and x[1] == 256), configs))
    print(f"Trying {len(configs)} configurations.")
    best, best_config = _autotune(configs, function)
    print("Found best_config ", best_config, " with time ", best, " for key ", key)
    BEST_CONFIGS[key] = best_config
    _save_best_configs(BEST_CONFIGS)
    return best_config, True

lib = torch.library.Library("custom_int_mm", "FRAGMENT")
lib.define("int_mm_dequant(Tensor a, Tensor b, Tensor scalar1, Tensor scalar2, ScalarType out_dtype) -> Tensor")

# All that's needed for torch.compile support
@torch.library.impl(lib, "int_mm_dequant", "Meta")
def _int_mm_dequant_meta(a, b, scalar1, scalar2, out_dtype):
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    return torch.empty((M, N), device=a.device, dtype=torch.bfloat16)


# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
@torch.library.impl(lib, "int_mm_dequant", "CUDA")
def _int_mm_dequant(a, b, scalar1, scalar2, out_dtype):
    # b = b.contiguous()
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.transpose(0, 1).is_contiguous(), "Matrix B must be transpose contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    scalar1 = scalar1.expand_as(c)
    scalar2 = scalar2.expand_as(c)
    assert scalar1.dim() == 2
    assert scalar2.dim() == 2
    assert out_dtype == torch.bfloat16
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    def matmul_kernel(a, b, c, scalar1, scalar2, BLOCK_M, BLOCK_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps):
        matmul_kernel_with_block_pointers[grid](
            a, b, c, scalar1, scalar2,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scalar1.stride(0), scalar1.stride(1),
            scalar2.stride(0), scalar2.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )

    import functools
    partial_matmul_kernel = functools.partial(matmul_kernel, a, b, c, scalar1, scalar2)

    best_config, first_time = _find_config([a, b, c, scalar1, scalar2], partial_matmul_kernel)
    partial_matmul_kernel(*best_config)

    return c
