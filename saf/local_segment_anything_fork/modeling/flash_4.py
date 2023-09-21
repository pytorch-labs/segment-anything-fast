"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import torch

import triton
import triton.language as tl


class _WipFlash2Library:
    lib = torch.library.Library("wipflash2", "DEF")
    ops_table: dict[tuple[str, str], callable] = {}

    @classmethod
    def registerOp(cls, op_key, full_schema, op_impl, dispatch_key):
        print("cls.ops_table: ", cls.ops_table)
        if (op_key, dispatch_key) not in cls.ops_table:
            if (op_key, "CUDA") not in cls.ops_table:
                cls.lib.define(full_schema)
            cls.lib.impl("wipflash2::" + op_key, op_impl, dispatch_key)
            cls.ops_table[(op_key, dispatch_key)] = op_impl
        return cls.ops_table[(op_key, dispatch_key)]


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _fwd_kernel(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qh, stride_qm,
    stride_kh, stride_kn,
    stride_vh, stride_vk,
    stride_oh, stride_om,
    stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
#    **META):
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
):
    # BLOCK_M = META['BLOCK_M']
    # BLOCK_N = META['BLOCK_N']
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(1, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr) #, boundary_check=(1, 0), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_mask = tl.arange(0, BLOCK_N)
    b_ptr_offsets_m = tl.arange(0, BLOCK_M)
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr) #, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr) #, boundary_check=(1, 0), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        qk += tl.dot(q, k, out_dtype=tl.float16) # * qk_scale).to(tl.float16)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        # TODO: Load this whole thing in broadcastable shape
        b_offset = off_hz * stride_b0h
        b_ptr_offsets_n_0 = (start_n + b_mask) // BIAS_LAST_SIZE
        b_ptr_offsets_n_1 = ((start_n + b_mask) % BIAS_LAST_SIZE) + BIAS_LAST_SIZE
        qk += tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + b_ptr_offsets_n_0[None, :], mask=((start_n + b_mask) < ((B0_NUMEL - 4) * (B0_NUMEL - 4)))[None, :], other=float('-inf'))
        qk += tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :], mask=((start_n + b_mask) < ((B0_NUMEL - 4) * (B0_NUMEL - 4)))[None, :], other=float('-inf'))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))

def _attention_rel_h_rel_w_kernel(q, k, v, rel_h_w, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)

    BLOCK_M = 128

    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3

    BLOCK_N = 64
    num_stages = 4

    num_warps = 4

    BLOCK_M = 64 # 128
    BLOCK_N = 64
    num_warps = 4

    # Auto tune this number
    num_stages = 1

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    # print("q.shape[0] * q.shape[1]: ", q.shape[0] * q.shape[1])
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
    assert P_SEQ == 0
    # assert rel_h.stride(0) == rel_w.stride(0)
    # assert rel_h.stride(1) == rel_w.stride(1)
    # assert rel_h.stride(2) == rel_w.stride(2)
    # assert rel_h.stride(3) == rel_w.stride(3)
    # assert rel_h.size(-1)  == rel_w.size(-1)
    b = rel_h_w
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert b.is_contiguous()
    _fwd_kernel[grid](
        q, k, v,
        b,
        sm_scale,
        o,
        q.stride(1), q.stride(2),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        o.stride(1), o.stride(2),
        b.stride(1), b.stride(2),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        P_SEQ,
        BIAS_LAST_SIZE=((b.size(-1) - 4) // 2),
        B0_NUMEL=b.size(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=num_stages)

    return o

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vk, stride_vn,
    stride_oh, stride_om, stride_on,
    stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
#    **META):
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
):
    # BLOCK_M = META['BLOCK_M']
    # BLOCK_N = META['BLOCK_N']
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr) #, boundary_check=(1, 0), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_hz * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) % BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :])
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr) #, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr) #, boundary_check=(1, 0), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        qk += tl.dot(q, k, out_dtype=tl.float16) # * qk_scale).to(tl.float16)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + start_n // BLOCK_N)
        qk += (b0 + b1)

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


def _autotune(configs, function):
    import torch.utils.benchmark as benchmark
    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
            t0 = benchmark.Timer(
                stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
            )
        except:
            return None
        return t0.blocked_autorange().mean * 1e6

    best = None
    best_config = None
    for config in configs:
        BLOCK_M, BLOCK_N, num_warps, num_stages = config
        t_config = benchmark_torch_function_in_microseconds(function, BLOCK_M, BLOCK_N, num_warps, num_stages)
        if t_config is not None:
            if best is not None:
                if t_config < best:
                    best = t_config
                    best_config = config
            else:
                best = t_config
                best_config = config
        print(str(config), " :", str(t_config))
    return best, best_config

BEST_CONFIGS = {}
BEST_CONFIGS[(torch.Size([60, 12, 4096, 64]), torch.Size([60, 12, 4096, 64]), torch.Size([60, 12, 4096, 64]), torch.Size([60, 12, 4096, 128]), torch.Size([60, 12, 4096, 64]), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (6291456, 524288, 128, 1), (3145728, 262144, 64, 1))] = (64, 64, 4, 2)
BEST_CONFIGS[(torch.Size([20, 12, 4096, 64]), torch.Size([20, 12, 4096, 64]), torch.Size([20, 12, 4096, 64]), torch.Size([20, 12, 4096, 128]), torch.Size([20, 12, 4096, 64]), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (6291456, 524288, 128, 1), (3145728, 262144, 64, 1))] = (64, 64, 4, 2)
BEST_CONFIGS[(torch.Size([ 1, 12, 4096, 64]), torch.Size([ 1, 12, 4096, 64]), torch.Size([ 1, 12, 4096, 64]), torch.Size([ 1, 12, 4096, 128]), torch.Size([ 1, 12, 4096, 64]), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (6291456, 524288, 128, 1), (3145728, 262144, 64, 1))] = (64, 128, 4, 2)

BEST_CONFIGS[(torch.Size([1, 16, 4096, 128]), torch.Size([1, 16, 4096, 128]), torch.Size([1, 16, 4096, 128]), torch.Size([1, 16, 4096, 128]), torch.Size([1, 16, 4096, 128]), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1))] = (128, 64, 8, 3)
BEST_CONFIGS[(torch.Size([100, 16, 4096, 128]), torch.Size([100, 16, 4096, 128]), torch.Size([100, 16, 4096, 128]), torch.Size([100, 16, 4096, 128]), torch.Size([100, 16, 4096, 128]), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1))] = (128, 64, 8, 3)
BEST_CONFIGS[(torch.Size([50, 16, 4096, 128]), torch.Size([50, 16, 4096, 128]), torch.Size([50, 16, 4096, 128]), torch.Size([50, 16, 4096, 128]), torch.Size([50, 16, 4096, 128]), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1))] = (128, 64, 8, 3)
BEST_CONFIGS[(torch.Size([40, 16, 4096, 128]), torch.Size([40, 16, 4096, 128]), torch.Size([40, 16, 4096, 128]), torch.Size([40, 16, 4096, 128]), torch.Size([40, 16, 4096, 128]), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1))] = (128, 64, 8, 3)
BEST_CONFIGS[(torch.Size([20, 16, 4096, 128]), torch.Size([20, 16, 4096, 128]), torch.Size([20, 16, 4096, 128]), torch.Size([20, 16, 4096, 128]), torch.Size([20, 16, 4096, 128]), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1), (8388608, 524288, 128, 1))] = (128, 64, 8, 3)

BEST_CONFIGS[(torch.Size([50, 12, 4096, 64]), torch.Size([50, 12, 4096, 64]), torch.Size([50, 12, 4096, 64]), torch.Size([50, 12, 4096, 128]), torch.Size([50, 12, 4096, 64]), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (6291456, 524288, 128, 1), (3145728, 262144, 64, 1))] = (64, 64, 4, 2)
BEST_CONFIGS[(torch.Size([100, 12, 4096, 64]), torch.Size([100, 12, 4096, 64]), torch.Size([100, 12, 4096, 64]), torch.Size([100, 12, 4096, 128]), torch.Size([100, 12, 4096, 64]), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (6291456, 524288, 128, 1), (3145728, 262144, 64, 1))] = (64, 64, 4, 2)

BEST_CONFIGS[(torch.Size([128, 12, 4096, 64]), torch.Size([128, 12, 4096, 64]), torch.Size([128, 12, 4096, 64]), torch.Size([128, 12, 4096, 128]), torch.Size([128, 12, 4096, 64]), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (3145728, 262144, 64, 1), (6291456, 524288, 128, 1), (3145728, 262144, 64, 1))] = (64, 64, 4, 2)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 num_warps,
                                                 num_stages):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert q.size() == k.size()
    assert q.size() == v.size()
    assert q.size(-2) == rel_h_w.size(-2)
    assert rel_h_w.size(-1) == 128
    # assert rel_h_w.size(-1) == 2 * BLOCK_N

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    # print("q.shape[0] * q.shape[1]: ", q.shape[0] * q.shape[1])
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
    assert P_SEQ == 0
    # assert rel_h.stride(0) == rel_w.stride(0)
    # assert rel_h.stride(1) == rel_w.stride(1)
    # assert rel_h.stride(2) == rel_w.stride(2)
    # assert rel_h.stride(3) == rel_w.stride(3)
    # assert rel_h.size(-1)  == rel_w.size(-1)
    b = rel_h_w
    # assert q.is_contiguous(), str(q.stride())
    # assert k.is_contiguous(), str(k.stride())
    # assert v.is_contiguous(), str(v.stride())
    # assert o.is_contiguous(), str(o.stride())
    assert b.is_contiguous(), str(b.stride())
    _fwd_kernel_aligned[grid](
        q, k, v,
        b,
        sm_scale,
        o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        b.stride(1), b.stride(2),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        P_SEQ,
        BIAS_LAST_SIZE=(b.size(-1) // 2),
        B0_NUMEL=b.size(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=num_stages)

def _attention_rel_h_rel_w_kernel_aligned(q, k, v, rel_h_w, sm_scale):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q, memory_format=torch.contiguous_format)

    key = (q.size(),   k.size(),   v.size(),   rel_h_w.size(),   o.size(),
           q.stride(), k.stride(), v.stride(), rel_h_w.stride(), o.stride())
    if key not in BEST_CONFIGS:
        print("key ", key, " not found. Running autotune")
        import functools
        import itertools
        configs = []
        for (BLOCK_M, BLOCK_N, num_warps) in itertools.product([64, 128], [64, 128], [1, 2, 4, 8]):
            for num_stages in range(1, num_warps + 1):
                configs.append((BLOCK_M, BLOCK_N, num_warps, num_stages))
        print("all configs len: ", len(configs))
        best, best_config = _autotune(configs, functools.partial(_attention_rel_h_rel_w_kernel_aligned_device,
                                                                 q, k, v, rel_h_w, sm_scale, o))
        BEST_CONFIGS[key] = best_config
        print("Found best_config ", best_config,
              " with time ", best, " for key ", key)
    best_config = BEST_CONFIGS[key]
    if best_config is None:
        return torch.tensor([])

    _attention_rel_h_rel_w_kernel_aligned_device(q,
                                                 k,
                                                 v,
                                                 rel_h_w,
                                                 sm_scale,
                                                 o,
                                                 best_config[0],
                                                 best_config[1],
                                                 best_config[2],
                                                 best_config[3])

    return o

def _attention_rel_h_rel_w(q_, k_, v_, rel_h_, rel_w_):
    """
    Implements SDPA but bias is addition of (rel_h + rel_w).view(..., rel_h.size(-2) * rel_w.size(-1))
    """

    import math
    sm_scale = 1. / math.sqrt(q_.size(-1))
    q_size_2_padded = (((q_.size(-2) + 256 - 1) // 256) * 256) - q_.size(-2)
    if q_size_2_padded == 0 and q_.size(-1) == 64:
        # print("USING ALIGNED")
        rel_h_w = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
        o = torch.ops.wipflash2.mah_flash_aligned(q_, k_, v_, rel_h_w, sm_scale)
        if o.numel() > 0:
            return o
    if q_size_2_padded == 0 and q_.size(-1) == 80:
        # print("USING ALIGNED")
        q = torch.nn.functional.pad(q_, (0, 128 - 80, 0, 0), "constant", 0) #.contiguous()
        k = torch.nn.functional.pad(k_, (0, 128 - 80, 0, 0), "constant", 0) #.contiguous()
        v = torch.nn.functional.pad(v_, (0, 128 - 80, 0, 0), "constant", 0) #.contiguous()
        rel_h_w = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
        o = torch.ops.wipflash2.mah_flash_aligned(q, k, v, rel_h_w, sm_scale)
        if o.numel() > 0:
            return o[:, :, :, :80] #.contiguous()
    attn_bias = (rel_h_ + rel_w_).view(q_.size(0), q_.size(1), rel_h_.size(2), rel_h_.size(3) * rel_w_.size(4))
    return torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_bias)
    print("USING NOT ALIGNED")
    q = torch.nn.functional.pad(q_, (0, 0, 0, q_size_2_padded), "constant", 0).contiguous()
    k = torch.nn.functional.pad(k_, (0, 0, 0, q_size_2_padded), "constant", 0).contiguous()
    v = torch.nn.functional.pad(v_, (0, 0, 0, q_size_2_padded), "constant", 0).contiguous()

    # rel_h = torch.nn.functional.pad(rel_h_.squeeze(-1), (0, 2, 0, q_size_2_padded), "constant", float("-inf"))
    # rel_w = torch.nn.functional.pad(rel_w_.squeeze(-2), (0, 2, 0, q_size_2_padded), "constant", float("-inf"))
    rel_h_w_ = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
    rel_h_w = torch.nn.functional.pad(rel_h_w_, (0, 4, 0, q_size_2_padded), "constant", float("-inf"))

    # o = _attention_rel_h_rel_w_kernel(q, k, v, rel_h_w, sm_scale)
    o = torch.ops.wipflash2.mah_flash(q, k, v, rel_h_w, sm_scale)
    return o[:, :, :q_.size(-2), :].contiguous()


_WipFlash2Library.registerOp(
    "mah_flash",
    "mah_flash(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor",
    _attention_rel_h_rel_w_kernel,
    "CUDA",
)


def _attention_rel_h_rel_w_kernel_meta(q_, k_, v_, rel_h_w, sm_scale):
    torch._check(q_.dim() == 4, f"Ugh wtf q is {q_.dim()}")
    torch._check(k_.dim() == 4, f"Ugh wtf k is {k_.dim()}")
    torch._check(v_.dim() == 4, f"Ugh wtf v is {v_.dim()}")

    torch._check(q_.is_contiguous() == 4, f"Ugh wtf q strdies is {q_.stride()}")
    torch._check(k_.is_contiguous() == 4, f"Ugh wtf k strdies is {k_.stride()}")
    torch._check(v_.is_contiguous() == 4, f"Ugh wtf v strdies is {v_.stride()}")
    return q_


_WipFlash2Library.registerOp(
    "mah_flash",
    "mah_flash(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor",
    _attention_rel_h_rel_w_kernel_meta,
    "Meta",
)

_WipFlash2Library.registerOp(
    "mah_flash_aligned",
    "mah_flash_aligned(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor",
    _attention_rel_h_rel_w_kernel_aligned,
    "CUDA",
)


def _attention_rel_h_rel_w_kernel_aligned_meta(q_, k_, v_, rel_h_w, sm_scale):
    return q_.contiguous()


_WipFlash2Library.registerOp(
    "mah_flash_aligned",
    "mah_flash_aligned(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor",
    _attention_rel_h_rel_w_kernel_aligned_meta,
    "Meta",
)
