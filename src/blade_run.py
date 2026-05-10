"""
blade_run.py - BLADE CLI entry point

Usage:
    python blade_run.py --prompt "Solve: 2x + 5 = 13\n"
    python blade_run.py --prompt "..." --causal
    python blade_run.py --prompt "..." --causal --save_diagnostics diag.json
"""

import json
import argparse
import yaml
import torch

from blade_generate import load_model, generate, causal_forward
from blade_causal import load as load_causal, plot_heatmap, print_causal_edges
from blade_mask import build_blade_mask


def load_config(path="blade_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="BLADE inference")
    p.add_argument("--config",           type=str,   default="./src/blade_config.yaml")
    p.add_argument("--model_path",       type=str,   default=None)
    p.add_argument("--prompt",           type=str,   required=True)
    p.add_argument("--max_tokens",       type=int,   default=None)
    p.add_argument("--beta",             type=float, default=None)
    p.add_argument("--seq_len",          type=int,   default=None,
                   help="Override seq_len (must be divisible by 8)")
    p.add_argument("--causal",           action="store_true",
                   help="Run causal forward and analysis after generation")
    p.add_argument("--save_diagnostics", type=str,   default="")
    p.add_argument("--causal_out",       type=str,   default="causal.png")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    model_dir  = args.model_path or cfg["model"]["output_dir"]
    beta       = args.beta       or cfg["generate"]["beta"]
    max_tokens = args.max_tokens or cfg["generate"]["max_tokens"]
    seq_len    = args.seq_len    or cfg["generate"]["seq_len"]
    num_blocks = seq_len // 8

    save_diag = args.save_diagnostics
    if args.causal and not save_diag:
        save_diag = "diag.json"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    print(f"[Init] device={device}  dtype={dtype}  model={model_dir}")
    print(f"[Init] seq_len={seq_len}  num_blocks={num_blocks}  block_size=8")

    model, tokenizer, mask_token_id, pad_token_id = load_model(
        model_dir, device, dtype
    )

    # ── Round 1: generation ───────────────────────────────────────────────────
    text, stats, draft = generate(
        model, tokenizer, mask_token_id, pad_token_id,
        prompt     = args.prompt,
        max_tokens = max_tokens,
        seq_len    = seq_len,
        num_blocks = num_blocks,
        beta       = beta,
        device     = device,
        dtype      = dtype,
    )

    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")
    print(f"[Stats] tokens={stats['tokens']}  time={stats['time']}s  "
          f"avg_H={stats['avg_H']}")

    # ── Round 2: causal forward (optional) ───────────────────────────────────
    if args.causal:
        print(f"\n[Causal] running causal forward...")
        attn_bias  = build_blade_mask(seq_len, num_blocks, device, dtype)
        prompt_ids = tokenizer.encode(
            args.prompt.replace('\\n', '\n').replace('\\t', '\t'),
            add_special_tokens=False,
        )
        final_tokens, a_blk, diag = causal_forward(
            model, draft, seq_len, num_blocks,
            mask_token_id, pad_token_id,
            attn_bias, beta, len(prompt_ids),
            device, dtype,
        )

        if save_diag:
            bs = seq_len // num_blocks
            block_texts = {
                i: tokenizer.decode(
                    final_tokens[i*bs:(i+1)*bs], skip_special_tokens=False
                )
                for i in range(num_blocks)
            }
            with open(save_diag, "w") as f:
                json.dump({
                    "prompt":      args.prompt,
                    "beta":        beta,
                    "seq_len":     seq_len,
                    "num_blocks":  num_blocks,
                    "block_size":  bs,
                    "draft":       draft,
                    "causal":      diag,
                    "a_blk":       a_blk[0].tolist(),
                    "block_texts": block_texts,
                    "generated":   text,
                }, f, indent=2, ensure_ascii=False)
            print(f"[Save] diagnostics → {save_diag}")

        causal_cfg = cfg["causal"]
        a, data    = load_causal(save_diag, block_size=causal_cfg["block_size"])
        plot_heatmap(a, args.causal_out)
        print_causal_edges(a, data,
                           min_gap=causal_cfg["min_gap"],
                           topk=causal_cfg["topk"])


if __name__ == "__main__":
    main()