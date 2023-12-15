import torch
import itertools
from segment_anything_fast.flash_4 import _attention_rel_h_rel_w

def test_op(batch, head, seq_len, hidden_dim, dtype):
    import math

    sm_scale = 1.0 / math.sqrt(hidden_dim)
    device = "cuda"
    torch.manual_seed(20)
    q = torch.empty(
        (batch, head, seq_len, hidden_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    k = torch.empty(
        (batch, head, seq_len, hidden_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    v = torch.empty(
        (batch, head, seq_len, hidden_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    w = int((seq_len) ** 0.5)
    assert w * w == seq_len, "seq_len must be a perfect square"

    rel_h = torch.empty(
        (batch, head, seq_len, w, 1), dtype=dtype, device=device
    ).normal_(mean=0, std=0.5)
    rel_w = torch.empty(
        (batch, head, seq_len, 1, w), dtype=dtype, device=device
    ).normal_(mean=0, std=0.5)

    tri_out = _attention_rel_h_rel_w(q, k, v, rel_h, rel_w)
    # reference implementation
    attn_bias = (rel_h + rel_w).view(
        q.size(0), q.size(1), rel_h.size(2), rel_h.size(3) * rel_w.size(4)
    )
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias
    )

    torch.testing.assert_close(ref_out, tri_out, rtol=1e-3, atol=1e-3)

for batch, (head, seq_len), dtype in itertools.product([1, 8], [(16, 80), (12, 64)], [torch.float16, torch.bfloat16]):
    print(f"batch: {batch} head: {head} seq_len: {seq_len} dtype: {dtype}")
    test_op(batch, head, 4096, seq_len, dtype)
