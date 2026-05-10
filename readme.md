# BLADE-LM

[Hugging Face](https://huggingface.co/Hengzongshu/BLADE-LM)

[中文](readme_zh.md) | English

**B**lock-Level **A**utoregressive **D**iffusion with Causal Discov**E**ry

BLADE is a hybrid language model that combines autoregressive (AR) generation with a diffusion-based think segment, enabling real-time causal discovery between token blocks during inference.

---

## How It Works

BLADE processes sequences in two parallel segments:

```
Input:  [ clean segment | think segment ]
         real tokens       all [MASK]
```

A fixed 4-quadrant attention mask governs how the two segments interact:

| Quadrant | Direction | Rule |
|---|---|---|
| Top-Left | clean → clean | AR causal (lower-triangular) |
| Top-Right | clean → think | clean[i] attends think[j < i] |
| Bottom-Left | think → clean | think[i] attends clean[j < i] |
| Bottom-Right | think → think | intra-block bidirectional only |

At inference, a two-round generation process activates the causal signal:

- **Round 1**: draft the full sequence token-by-token using a mixed AR/Diff decision
- **Round 2**: single forward pass with the draft as input — future clean tokens activate K_diff via the top-right quadrant, enabling causal analysis

The `beta` parameter controls the AR/Diff mix ratio (0 = pure Diff, 1 = pure AR).

---

## Causal Discovery

After generation, BLADE can produce a block-level causal graph showing which past blocks influenced which future blocks. This graph is derived from the top-right attention weights extracted during Round 2.

Example output (math reasoning):

```
[1] block 1 → block 33  gap=32  strength=0.0017
  cause  j=1:  ' + 5 = 13.'
  effect i=33: '2x + 5 = 1'

[2] block 2 → block 21  gap=19  strength=0.0013
  cause  j=2:  ' Show your work. To solve the equation'
  effect i=21: '4\n   \]\n\n3. **'
```

The model adapts its causal depth to the complexity of the task — simple queries produce shallow linear graphs, while multi-step reasoning produces richer long-range dependencies.

---

## Installation

```bash
git clone https://github.com/xiaziye/BLADE-LM.git
cd BLADE-LM
pip install -r requirements.txt
```

Install PyTorch separately to match your CUDA version:
```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only
pip install torch
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0
- `bitsandbytes` — optional, enables 8-bit optimizer for training
- `matplotlib` — optional, required for causal heatmap
- `datasets` — optional, required for HuggingFace data sources

---

## Quick Start

### Generation

```bash
# Basic generation
python blade_run.py --prompt "Solve: 2x + 5 = 13. Show your work.\n"

# With causal graph analysis
python blade_run.py \
    --prompt "Solve: 2x + 5 = 13. Show your work.\n" \
    --causal \
    --causal_out causal.png
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--prompt` | — | Input prompt |
| `--beta` | 0.6 | AR/Diff mix ratio (0.49–0.90 recommended) |
| `--max_tokens` | 200 | Maximum tokens to generate |
| `--causal` | false | Run causal graph analysis after generation |
| `--causal_out` | causal.png | Causal heatmap output path |
| `--save_diagnostics` | — | Save full diagnostics to JSON |

All defaults are set in `blade_config.yaml`.

---

## Configuration

Edit `blade_config.yaml` to configure the model, data, training, and inference:

```yaml
model:
  base_model: "./base_model"    # path to base model (e.g. Qwen2.5-0.5B)
  output_dir: "./BLADE-v1"      # trained model save path

generate:
  beta:       0.6               # AR/Diff mix ratio
  max_tokens: 200

data:
  sources:
    - name_tag:   "my_data"
      path:       "./data/train.jsonl"
      text_field: "problem,solution"   # concat two fields
      weight:     1
```

### Data Format

Local JSONL — one JSON object per line:
```json
{"problem": "Solve 2x + 5 = 13", "solution": "x = 4"}
{"text": "Any plain text field also works"}
```

HuggingFace datasets are also supported directly via `hf_name`.  
Multiple sources with different weights can be mixed freely.

---

## Training

```bash
# Train with default config
python blade_train.py

# Resume from checkpoint
python blade_train.py --resume ./ckpt/step20000.pt

# Use a custom config
python blade_train.py --config my_config.yaml
```

Training loss components:

| Loss | Weight | Description |
|---|---|---|
| `l_diff` | 1.0 | Diffusion loss on think segment |
| `l_ar` | `alpha` | AR loss on clean segment |
| `L_intra` | `lambda_intra` | Intra-block token diversity |

---

## Causal Analysis

Run causal analysis on saved diagnostics:

```bash
python blade_causal.py --json diag.json --out causal.png --min_gap 3 --topk 15
```

The heatmap visualizes the N×N block attention matrix from the top-right quadrant. Non-zero values at position (i, j) indicate that clean block i depends causally on think block j (j < i).


---

## Citation

```bibtex
@misc{blade2025,
  title  = {BLADE-LM: Block-Level Autoregressive Diffusion with Causal Discovery},
  author = {Xia Ziye},
  year   = {2025},
  url    = {https://github.com/xiaziye/BLADE-LM}
}
```
