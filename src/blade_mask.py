"""
blade_mask.py - BLADE 4-quadrant attention mask

Quadrant rules (Q x K):
  Top-Left     clean→clean  AR causal (lower-triangular)
  Top-Right    clean→think  MOBC reversed: clean[i] attends think[j < i]
  Bottom-Left  think→clean  MOBC: think[i] attends clean[j < i]
  Bottom-Right think→think  intra-block bidirectional, cross-block masked
"""

import torch


def build_blade_mask(
    seq_len:    int,
    num_blocks: int,
    device:     torch.device,
    dtype,
) -> torch.Tensor:
    """Build BLADE fixed 4-quadrant attention mask.
    Returns additive bias of shape (1, 1, 2L, 2L), values 0 or -inf.
    """
    L   = seq_len
    bs  = seq_len // num_blocks
    TL  = 2 * L
    NEG = float('-inf')

    bias = torch.zeros(1, 1, TL, TL, device=device, dtype=dtype)

    idx      = torch.arange(TL, device=device)
    is_think = idx >= L
    pos      = idx % L
    blk      = pos // bs

    is_tq = is_think.unsqueeze(1)
    is_tk = is_think.unsqueeze(0)
    blk_q = blk.unsqueeze(1)
    blk_k = blk.unsqueeze(0)
    pos_q = pos.unsqueeze(1)
    pos_k = pos.unsqueeze(0)

    # Top-Left: clean→clean, AR lower-triangular
    cc = (~is_tq) & (~is_tk) & (pos_k > pos_q)
    bias.masked_fill_(cc.unsqueeze(0).unsqueeze(0), NEG)

    # Top-Right: clean→think, clean[i] attends think[j < i]
    ct = (~is_tq) & is_tk & (blk_k >= blk_q)
    bias.masked_fill_(ct.unsqueeze(0).unsqueeze(0), NEG)

    # Bottom-Left: think→clean, think[i] attends clean[j < i]
    tc = is_tq & (~is_tk) & (blk_k >= blk_q)
    bias.masked_fill_(tc.unsqueeze(0).unsqueeze(0), NEG)

    # Bottom-Right: think→think, intra-block only
    tt = is_tq & is_tk & (blk_q != blk_k)
    bias.masked_fill_(tt.unsqueeze(0).unsqueeze(0), NEG)

    return bias


def get_block_attention(
    attn_weights: torch.Tensor,
    seq_len:      int,
    num_blocks:   int,
) -> torch.Tensor:
    """Extract top-right (clean→think) block-level attention weights.

    Args:
        attn_weights: (B, H, 2L, 2L) attention weights from one layer
    Returns:
        (B, N, N) block-averaged attention, [b, i, j] = clean[i] → think[j]
        Non-zero only for j < i (BLADE mask guarantee)
    """
    L  = seq_len
    bs = seq_len // num_blocks
    N  = num_blocks

    a    = attn_weights.mean(dim=1)          # (B, 2L, 2L)
    a_ru = a[:, :L, L:]                      # (B, L, L)

    B = a_ru.shape[0]
    a_ru = a_ru.view(B, N, bs, N, bs)
    return a_ru.mean(dim=(2, 4))             # (B, N, N)


if __name__ == "__main__":
    L, N = 24, 3
    bs   = L // N
    m    = build_blade_mask(L, N, torch.device("cpu"), torch.float32)
    vis  = (m[0, 0] == float("-inf")).int()
    print(f"Mask shape: {m.shape}")
    print(f"seq_len={L}, num_blocks={N}, block_size={bs}  (1=masked, 0=open)\n")
    print("     ", " ".join(f"{i:2d}" for i in range(2 * L)))
    for i in range(2 * L):
        row    = " ".join(f"{vis[i,j].item():2d}" for j in range(2 * L))
        marker = "T" if i >= L else "C"
        print(f"{marker}{i%L:2d}: {row}")
