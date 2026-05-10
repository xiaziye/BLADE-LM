"""
blade_generate.py - BLADE core generation functions (importable)

Two-round generation:
  Round 1: token-by-token draft with fixed beta mix
  Round 2: single eager forward, activates K_diff via future Q_clean
"""

import json
import time
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from blade_mask import build_blade_mask, get_block_attention


def load_model(model_dir, device, dtype, attn_impl="eager"):
    tokenizer     = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    _fim          = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    _unk          = tokenizer.unk_token_id or -1
    mask_token_id = _fim if _fim not in (None, _unk) else tokenizer.eos_token_id
    pad_token_id  = tokenizer.pad_token_id or tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype,
        trust_remote_code=True, attn_implementation=attn_impl,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    return model, tokenizer, mask_token_id, pad_token_id


def build_input(confirmed, seq_len, mask_token_id, pad_token_id, device):
    L     = seq_len
    clean = confirmed[:L] + [pad_token_id] * (L - len(confirmed))
    return torch.tensor(
        clean + [mask_token_id] * L,
        dtype=torch.long, device=device
    ).unsqueeze(0)


@torch.no_grad()
def draft_generate(
    model, prompt_ids, target_len,
    mask_token_id, pad_token_id,
    attn_bias, beta, device, dtype,
) -> List[int]:
    seq_len   = attn_bias.shape[-1] // 2   # recover seq_len from mask shape
    confirmed = list(prompt_ids)
    while len(confirmed) < target_len:
        x   = build_input(confirmed, seq_len, mask_token_id, pad_token_id, device)
        with torch.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            out = model(input_ids=x, attention_mask=attn_bias)
        pos    = len(confirmed)
        ar_l   = out.logits[0, pos - 1, :].float()
        diff_l = out.logits[0, seq_len + pos, :].float()
        tok    = (beta * ar_l + (1.0 - beta) * diff_l).argmax().item()
        confirmed.append(tok)
        if tok == model.config.eos_token_id:
            break
    return confirmed


@torch.no_grad()
def causal_forward(
    model, draft, seq_len, num_blocks,
    mask_token_id, pad_token_id,
    attn_bias, beta, prompt_len,
    device, dtype,
) -> Tuple[List[int], torch.Tensor, List[Dict]]:
    L, N, bs = seq_len, num_blocks, seq_len // num_blocks
    x = build_input(draft, L, mask_token_id, pad_token_id, device)

    with torch.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
        out = model(
            input_ids=x, attention_mask=attn_bias,
            output_attentions=True,
        )

    logits = out.logits[0].float()
    a_blk  = get_block_attention(out.attentions[-1], L, N)

    final_tokens = list(draft[:prompt_len])
    diag         = []

    for pos in range(prompt_len, L):
        ar_l   = logits[pos - 1]
        diff_l = logits[L + pos]
        p_diff = F.softmax(diff_l, dim=-1)
        H      = -(p_diff * (p_diff + 1e-9).log()).sum().item()
        tok    = (beta * ar_l + (1.0 - beta) * diff_l).argmax().item()
        final_tokens.append(tok)
        diag.append({
            "pos":      pos,
            "tok":      tok,
            "H":        round(H, 4),
            "strength": round(a_blk[0, pos // bs].sum().item(), 6),
        })
        if tok == model.config.eos_token_id:
            break

    return final_tokens, a_blk, diag


@torch.no_grad()
def generate(
    model, tokenizer, mask_token_id, pad_token_id,
    prompt, max_tokens, seq_len, num_blocks, beta,
    device, dtype, save_diagnostics="",
):
    assert seq_len % 8 == 0, f"seq_len must be divisible by 8, got {seq_len}"
    assert num_blocks == seq_len // 8, f"num_blocks must equal seq_len // 8"
    L, N, bs = seq_len, num_blocks, seq_len // num_blocks
    prompt       = prompt.replace('\\n', '\n').replace('\\t', '\t')
    prompt_ids   = tokenizer.encode(prompt, add_special_tokens=False)

    # effective generation limit
    max_gen = min(max_tokens, seq_len - len(prompt_ids))
    if max_gen <= 0:
        raise ValueError(f"Prompt too long ({len(prompt_ids)} tokens) for seq_len={seq_len}")
    target_len = len(prompt_ids) + max_gen

    attn_bias    = build_blade_mask(L, N, device, dtype)

    # Round 1
    t0    = time.time()
    draft = draft_generate(
        model, prompt_ids, target_len,
        mask_token_id, pad_token_id,
        attn_bias, beta, device, dtype,
    )
    # pad draft to full seq_len for Round 2 forward
    while len(draft) < seq_len:
        draft.append(pad_token_id)
    t1 = time.time()

    # Round 2
    final_tokens, a_blk, diag = causal_forward(
        model, draft, L, N,
        mask_token_id, pad_token_id,
        attn_bias, beta, len(prompt_ids),
        device, dtype,
    )
    t2 = time.time()

    gen_ids = final_tokens[len(prompt_ids):][:max_tokens]
    text    = tokenizer.decode(gen_ids, skip_special_tokens=True)

    stats = {
        "tokens":      len(gen_ids),
        "time":        round(t2 - t0, 2),
        "draft_time":  round(t1 - t0, 2),
        "causal_time": round(t2 - t1, 2),
        "avg_H":       round(sum(d["H"] for d in diag) / max(len(diag), 1), 4),
        "avg_strength": round(sum(d["strength"] for d in diag) / max(len(diag), 1), 6),
    }
    return text, stats, draft