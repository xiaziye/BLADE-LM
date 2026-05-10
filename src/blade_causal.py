"""
blade_causal.py - BLADE causal graph analysis

Reads diagnostics from blade_generate.py (--save_diagnostics),
outputs a causal heatmap and top causal edges with block text.

Usage:
    python blade_causal.py --json diag.json --out causal.png
"""

import json
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load(json_path, block_size=8):
    with open(json_path) as f:
        data = json.load(f)
    a = np.array(data["a_blk"], dtype=np.float32)

    # Trim to valid blocks only
    diag = data.get("causal", [])
    last_pos     = max(d["pos"] for d in diag) if diag else a.shape[0] * block_size
    valid_blocks = min(last_pos // block_size + 1, a.shape[0])
    a = a[:valid_blocks, :valid_blocks]

    print(f"[Load] a_blk: {a.shape}  (trimmed to {valid_blocks} valid blocks)")
    return a, data


def plot_heatmap(a, out_path):
    if not HAS_MPL:
        print("[Skip] matplotlib not installed")
        return
    N = a.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(a, cmap="hot", aspect="auto",
                   vmin=0, vmax=max(a.max(), 1e-9))
    ax.set_xlabel("think block j")
    ax.set_ylabel("clean block i")
    ax.set_title(f"BLADE Causal Heatmap ({N} blocks)\n"
                 f"clean[i] ← think[j<i]")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Plot] saved: {out_path}")


def print_causal_edges(a, data, min_gap=3, topk=15):
    """Print top causal edges with block text content."""
    block_texts = data.get("block_texts", {})
    N = a.shape[0]

    edges = [
        (i, j, i - j, a[i, j])
        for i in range(N)
        for j in range(i)
        if i - j >= min_gap and a[i, j] > 1e-9
    ]
    edges.sort(key=lambda x: -x[3])

    print(f"\n{'='*60}")
    print(f" Top causal edges (gap >= {min_gap})")
    print(f"{'='*60}")
    for rank, (i, j, gap, s) in enumerate(edges[:topk]):
        ti = block_texts.get(str(i), "?")
        tj = block_texts.get(str(j), "?")
        print(f"\n[{rank+1}] block {j} → block {i}  gap={gap}  strength={s:.4f}")
        print(f"  cause  j={j}: {repr(tj)}")
        print(f"  effect i={i}: {repr(ti)}")


def parse_args():
    p = argparse.ArgumentParser(description="BLADE causal graph analysis")
    p.add_argument("--json",       type=str, default="diag.json")
    p.add_argument("--out",        type=str, default="causal.png")
    p.add_argument("--min_gap",    type=int, default=3)
    p.add_argument("--topk",       type=int, default=15)
    p.add_argument("--block_size", type=int, default=8)
    return p.parse_args()


def main():
    args   = parse_args()
    a, data = load(args.json, block_size=args.block_size)
    plot_heatmap(a, args.out)
    print_causal_edges(a, data, min_gap=args.min_gap, topk=args.topk)


if __name__ == "__main__":
    main()
