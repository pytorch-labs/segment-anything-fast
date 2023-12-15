import torch
from segment_anything_fast.flash_4 import _attention_rel_h_rel_w

def test_op(batch, head, seq_len, hidden_dim, dtype=torch.float16):
    import math

    sm_scale = 1.0 / math.sqrt(hidden_dim)
    device = "cuda"
    torch.manual_seed(20)
    q_ = torch.empty(
        (batch, head, seq_len, hidden_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    k_ = torch.empty(
        (batch, head, seq_len, hidden_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    v_ = torch.empty(
        (batch, head, seq_len, hidden_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    w = int((seq_len) ** 0.5)
    assert w * w == seq_len, "seq_len must be a perfect square"

    rel_h_ = torch.empty(
        (batch, head, seq_len, w, 1), dtype=dtype, device=device
    ).normal_(mean=0, std=0.5)
    rel_w_ = torch.empty(
        (batch, head, seq_len, 1, w), dtype=dtype, device=device
    ).normal_(mean=0, std=0.5)

    tri_out = _attention_rel_h_rel_w(q, k, v, rel_h, rel_w)
    # reference implementation
    attn_bias = (rel_h_ + rel_w_).view(
        q_.size(0), q_.size(1), rel_h_.size(2), rel_h_.size(3) * rel_w_.size(4)
    )
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=attn_bias
    )

    # compare
    print("max diff: ", (ref_out - tri_out).abs().max().item())
    print(
        torch.nn.functional.cosine_similarity(ref_out.ravel(), tri_out.ravel(), dim=-1)
    )
