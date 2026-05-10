"""
blade_train.py - BLADE training script

Loss:
    L = l_diff + alpha * l_ar + lambda_intra * L_intra

Usage:
    python blade_train.py --total_steps 40000
    python blade_train.py --resume ./ckpt/step20000.pt --total_steps 40000
"""

import os
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from blade_data import get_dataloader
from blade_mask import build_blade_mask


def load_config(path="blade_config.yaml"):
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = {}
    cfg["model_path"]      = raw["model"]["base_model"]
    cfg["ckpt_dir"]        = raw["train"]["ckpt_dir"]
    cfg["log_dir"]         = raw["train"]["log_dir"]
    cfg["seq_len"]         = raw["sequence"]["seq_len"]
    cfg["num_blocks"]      = raw["sequence"]["num_blocks"]
    cfg["cache_dir"]       = raw["data"].get("cache_dir", "./cache")
    cfg["data"]            = raw["data"]
    cfg.update(raw["train"])
    cfg.update(raw["loss"])
    return cfg


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_intra(hidden_states, seq_len, num_blocks):
    """Intra-block token diversity loss on think segment."""
    L  = seq_len
    bs = seq_len // num_blocks
    h  = hidden_states[:, L:, :]
    B, _, D = h.shape

    h   = h.view(B, num_blocks, bs, D)
    h   = F.normalize(h, dim=-1)
    sim = torch.matmul(h, h.transpose(-1, -2))   # (B, N, bs, bs)

    mask = torch.triu(
        torch.ones(bs, bs, device=h.device, dtype=torch.bool), diagonal=1
    )
    if mask.sum() == 0:
        return h.new_zeros(())
    return sim[..., mask].mean()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, cfg, final=False):
    name = "final.pt" if final else f"step{step}.pt"
    path = os.path.join(cfg["ckpt_dir"], name)
    obj  = {"step": step, "model": model.state_dict(), "config": cfg}
    if cfg.get("save_optimizer"):
        obj["optimizer"] = optimizer.state_dict()
        obj["scheduler"] = scheduler.state_dict()
    torch.save(obj, path)
    print(f"[Train] checkpoint saved: {path}")

    if not final:
        existing = sorted(
            Path(cfg["ckpt_dir"]).glob("step*.pt"),
            key=lambda p: int(p.stem.replace("step", ""))
        )
        for old in existing[:-cfg.get("keep_ckpts", 3)]:
            old.unlink()
            print(f"[Train] removed old checkpoint: {old}")


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[Train] device={device}  dtype={dtype}")

    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Model
    tokenizer     = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    _fim          = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    _unk          = tokenizer.unk_token_id or -1
    mask_token_id = _fim if _fim not in (None, _unk) else tokenizer.eos_token_id
    pad_token_id  = tokenizer.pad_token_id or tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=dtype,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    model.enable_input_require_grads()
    print(f"[Train] parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Data
    loader, _ = get_dataloader(cfg)

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95),
        )
        print("[Train] optimizer: AdamW 8-bit")
    except ImportError:
        optimizer = AdamW(
            model.parameters(), lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95),
        )
        print("[Train] optimizer: AdamW")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(cfg["total_steps"] - cfg["warmup_steps"], 1),
        eta_min=cfg["lr"] * 0.1,
    )

    # Mask
    L  = cfg["seq_len"]
    N  = cfg["num_blocks"]
    bs = L // N
    attn_bias = build_blade_mask(L, N, device, dtype)
    print(f"[Train] mask built  L={L}  N={N}  block_size={bs}")

    # Resume
    start_step = 0
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        print(f"[Train] resumed from step={start_step}")

    # Log
    log = {k: [] for k in ["step", "loss_total", "loss_diff",
                            "loss_ar", "loss_intra", "lr", "grad_norm"]}

    def recent(lst, n=50):
        return float(np.mean(lst[-n:])) if lst else 0.0

    def infinite_loader(ldr):
        while True:
            for batch in ldr:
                yield batch

    print(f"\n{'='*55}")
    print(f" BLADE Training")
    print(f" total_steps={cfg['total_steps']}  L={L}  N={N}  bs={bs}")
    print(f" alpha={cfg['alpha']}  lambda_intra={cfg['lambda_intra']}")
    print(f"{'='*55}\n")

    step       = start_step
    accum_step = 0
    accum_cache = []
    optimizer.zero_grad()

    for tokens_raw in infinite_loader(loader):
        if step >= cfg["total_steps"]:
            break

        # LR warmup
        if step < cfg["warmup_steps"]:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * (step + 1) / cfg["warmup_steps"]

        x_clean = tokens_raw.view(tokens_raw.shape[0], -1).to(device)
        x_think = torch.full_like(x_clean, mask_token_id)
        x_full  = torch.cat([x_clean, x_think], dim=1)
        B       = x_clean.shape[0]
        V       = model.config.vocab_size

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_bf16):
            out           = model(
                input_ids            = x_full,
                attention_mask       = attn_bias,
                output_hidden_states = True,
            )
            logits        = out.logits
            hidden_states = out.hidden_states[-1]

            # l_ar: AR loss on clean segment
            l_ar = F.cross_entropy(
                logits[:, :L-1, :].reshape(-1, V),
                x_clean[:, 1:].reshape(-1),
                ignore_index=pad_token_id,
            )

            # l_diff: diffusion loss on think segment
            valid = (x_clean != pad_token_id)
            if valid.sum() > 0:
                l_diff = F.cross_entropy(
                    logits[:, L:, :].reshape(-1, V)[valid.reshape(-1)],
                    x_clean.reshape(-1)[valid.reshape(-1)],
                )
            else:
                l_diff = logits.new_zeros(())

            # L_intra: intra-block diversity on think segment
            l_intra = loss_intra(hidden_states, L, N)

            loss = l_diff + cfg["alpha"] * l_ar + cfg["lambda_intra"] * l_intra

        (loss / cfg["grad_accum"]).backward()
        accum_step += 1
        accum_cache.append((loss.item(), l_diff.item(), l_ar.item(), l_intra.item()))

        if accum_step >= cfg["grad_accum"]:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["max_grad_norm"]
            ))
            optimizer.step()
            if step >= cfg["warmup_steps"]:
                scheduler.step()
            optimizer.zero_grad()
            accum_step = 0

            arr = np.array(accum_cache).mean(axis=0)
            accum_cache.clear()
            cur_lr = optimizer.param_groups[0]["lr"]

            log["step"].append(step)
            log["loss_total"].append(float(arr[0]))
            log["loss_diff"].append(float(arr[1]))
            log["loss_ar"].append(float(arr[2]))
            log["loss_intra"].append(float(arr[3]))
            log["lr"].append(cur_lr)
            log["grad_norm"].append(grad_norm)

            step += 1

            if step % cfg["log_every"] == 0:
                print(
                    f"[{step:6d}]  "
                    f"loss={recent(log['loss_total']):.4f}  "
                    f"diff={recent(log['loss_diff']):.4f}  "
                    f"ar={recent(log['loss_ar']):.4f}  "
                    f"intra={recent(log['loss_intra']):.4f}  "
                    f"|g|={recent(log['grad_norm']):.3f}  "
                    f"lr={cur_lr:.2e}"
                )

            if step % cfg["save_every"] == 0:
                save_checkpoint(model, optimizer, scheduler, step, cfg)

    save_checkpoint(model, optimizer, scheduler, step, cfg, final=True)

    log_path = os.path.join(cfg["log_dir"], "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[Train] log saved: {log_path}")

    print(f"\n{'='*55}")
    print(f" Final metrics (last 50 steps avg)")
    print(f"  loss_diff : {recent(log['loss_diff']):.4f}")
    print(f"  loss_ar   : {recent(log['loss_ar']):.4f}")
    print(f"{'='*55}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BLADE training")
    p.add_argument("--config", type=str, default="blade_config.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    if args.resume:
        cfg["resume"] = args.resume

    print("[Config]")
    for k, v in sorted(cfg.items()):
        print(f"  {k:25s} = {v}")
    print()

    train(cfg)


if __name__ == "__main__":
    main()
