"""
blade_data.py - BLADE data loading

Supports local JSONL files and HuggingFace datasets.
Data format: each sample should be a text string (problem + solution / Q&A / etc.)

Configure data sources in blade_config.yaml:

    data:
      sources:
        - path: "./data/my_dataset.jsonl"   # local JSONL
          text_field: "text"                # or "Q,A" to concat multiple fields
          weight: 3
        - hf_name: "math-ai/StackMathQA"   # HuggingFace dataset
          hf_split: "train"
          text_field: "Q,A"
          weight: 5
"""

import os
import json
import hashlib
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Generic packed text dataset. Supports local JSONL and HuggingFace sources."""

    def __init__(
        self,
        seq_len:     int,
        tokenizer,
        path:        str  = None,      # local JSONL path
        hf_name:     str  = None,      # HuggingFace dataset name
        hf_config:   str  = None,
        hf_split:    str  = "train",
        text_field:  str  = "text",    # field name, or "Q,A" to concat
        cache_dir:   str  = "./cache",
        max_samples: int  = None,
        max_load:    int  = 200000,
        name_tag:    str  = "dataset",
    ):
        self.seq_len      = seq_len
        self.tokenizer    = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._tokens: List[torch.Tensor] = []

        src       = path or hf_name or "unknown"
        cache_key = hashlib.md5(
            f"{name_tag}_{src}_{hf_config}_{hf_split}_{text_field}_{seq_len}_{max_load}".encode()
        ).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{name_tag}_{cache_key}.pt")

        if os.path.exists(cache_path):
            print(f"[{name_tag}] loading cache: {cache_path}")
            self._tokens = torch.load(cache_path, weights_only=False)
        else:
            texts = self._load(path, hf_name, hf_config, hf_split, text_field, max_load)
            token_lists = [t for t in (self._encode(s) for s in texts) if t]
            self._tokens = self._pack(token_lists)
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(self._tokens, cache_path)
            print(f"[{name_tag}] cache saved: {cache_path}")

        if max_samples:
            self._tokens = self._tokens[:max_samples]
        print(f"[{name_tag}] {len(self._tokens)} windows")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, path, hf_name, hf_config, hf_split, text_field, max_load):
        texts = []
        if hf_name:
            texts = self._load_hf(hf_name, hf_config, hf_split, text_field, max_load)
        elif path:
            texts = self._load_jsonl(path, text_field, max_load)
        print(f"  loaded {len(texts)} texts")
        return texts

    def _load_hf(self, hf_name, hf_config, hf_split, text_field, max_load):
        from datasets import load_dataset
        try:
            ds = load_dataset(hf_name, hf_config, split=hf_split) if hf_config \
                 else load_dataset(hf_name, split=hf_split)
        except Exception:
            ds = load_dataset(hf_name, hf_config, split=hf_split, streaming=True) if hf_config \
                 else load_dataset(hf_name, split=hf_split, streaming=True)

        texts = []
        for row in ds:
            t = self._extract(row, text_field)
            if len(t.strip()) > 50:
                texts.append(t)
            if len(texts) >= max_load:
                break
        return texts

    def _load_jsonl(self, path, text_field, max_load):
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    t   = self._extract(row, text_field)
                except json.JSONDecodeError:
                    t = line
                if len(t.strip()) > 50:
                    texts.append(t)
                if len(texts) >= max_load:
                    break
        return texts

    @staticmethod
    def _extract(row, text_field):
        """Support 'field' or 'field1,field2' (concatenated with newline)."""
        if "," in text_field:
            parts = [str(row.get(f.strip(), "")).strip() for f in text_field.split(",")]
            return "\n\n".join(p for p in parts if p)
        return str(row.get(text_field, ""))

    # ── Tokenization + packing ────────────────────────────────────────────────

    def _encode(self, text):
        if len(text.strip()) < 20:
            return []
        enc = self.tokenizer(
            text, max_length=self.seq_len,
            padding=False, truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0).tolist()

    def _pack(self, token_lists):
        """Greedy first-fit packing into fixed-length windows."""
        win, packed = [], []

        def flush():
            if win:
                pad = self.seq_len - len(win)
                packed.append(torch.tensor(
                    win + [self.pad_token_id] * pad, dtype=torch.long
                ))

        for toks in token_lists:
            toks = toks + [self.eos_token_id]
            if len(toks) > self.seq_len:
                toks = toks[:self.seq_len]
            if len(win) + len(toks) > self.seq_len:
                flush()
                win = []
            win.extend(toks)
            if len(win) == self.seq_len:
                flush()
                win = []

        flush()
        return packed

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        return self._tokens[idx].long()


# ── DataLoader builder ────────────────────────────────────────────────────────

def get_dataloader(cfg: dict):
    """Build a weighted mixed DataLoader from config.

    cfg["data"]["sources"] is a list of source dicts, each with:
        path / hf_name, text_field, weight, and optional hf_config, hf_split
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)

    seq_len   = cfg["seq_len"]
    cache_dir = cfg.get("cache_dir", "./cache")
    sources   = cfg.get("data", {}).get("sources", [])

    if not sources:
        raise ValueError("No data sources configured. Check blade_config.yaml.")

    datasets, weights = [], []
    for i, src in enumerate(sources):
        tag = src.get("name_tag", f"source_{i}")
        ds  = TextDataset(
            seq_len     = seq_len,
            tokenizer   = tokenizer,
            path        = src.get("path"),
            hf_name     = src.get("hf_name"),
            hf_config   = src.get("hf_config"),
            hf_split    = src.get("hf_split", "train"),
            text_field  = src.get("text_field", "text"),
            cache_dir   = cache_dir,
            max_samples = src.get("max_samples"),
            max_load    = src.get("max_load", 200000),
            name_tag    = tag,
        )
        datasets.append(ds)
        w = src.get("weight", 1)
        weights.extend([w / len(ds)] * len(ds))

    combined = ConcatDataset(datasets)
    sampler  = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
    loader   = DataLoader(
        combined,
        batch_size         = cfg.get("batch_size", 4),
        sampler            = sampler,
        num_workers        = cfg.get("num_workers", 4),
        pin_memory         = True,
        persistent_workers = cfg.get("num_workers", 4) > 0,
    )

    print(f"[Data] total windows: {sum(len(d) for d in datasets)}")
    return loader, tokenizer
