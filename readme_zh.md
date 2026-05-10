# BLADE-LM

[English](./README.md) | 中文

**B**lock-Level **A**utoregressive **D**iffusion with Causal Discov**E**ry

BLADE-LM 是一个混合语言模型，将自回归生成（AR）与扩散式 think 段相结合，能够在推理过程中实时发现 token 块之间的因果依赖关系。

---

## 工作原理

BLADE-LM 将输入序列拆分为两个并行段：

```
输入：[ clean 段 | think 段 ]
       真实 token    全部 [MASK]
```

固定的 4 象限 attention mask 控制两段之间的交互方式：

| 象限 | 方向 | 规则 |
|---|---|---|
| 左上 | clean → clean | AR 因果（下三角） |
| 右上 | clean → think | clean[i] 关注 think[j < i] |
| 左下 | think → clean | think[i] 关注 clean[j < i] |
| 右下 | think → think | 仅同块双向，跨块屏蔽 |

推理时采用两轮生成流程来激活因果信号：

- **第一轮**：使用 AR/Diff 混合决策逐 token 生成草稿序列
- **第二轮**：以草稿作为 clean 段输入，做一次完整 forward——未来 clean token 通过右上象限激活 K_diff，从而支持因果分析

参数 `beta` 控制 AR/Diff 混合比例（0 = 纯 Diff，1 = 纯 AR）。

---

## 因果发现

生成完成后，BLADE-LM 可以输出块级因果图，展示过去的块对未来块的影响关系。该图由第二轮 forward 中提取的右上象限 attention 权重得出。

示例输出（数学推理）：

```
[1] block 1 → block 33  gap=32  strength=0.0017
  原因 j=1:  ' + 5 = 13.'
  结果 i=33: '2x + 5 = 1'

[2] block 2 → block 21  gap=19  strength=0.0013
  原因 j=2:  ' Show your work. To solve the equation'
  结果 i=21: '4\n   \]\n\n3. **'
```

模型会根据任务复杂度自适应调整因果深度——简单问题生成浅层线性因果图，多步推理则产生更丰富的长程依赖关系。

---

## 安装

```bash
git clone https://github.com/xiaziye/BLADE-LM.git
cd BLADE-LM
pip install -r requirements.txt
```

根据你的 CUDA 版本单独安装 PyTorch：

```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# 仅 CPU
pip install torch
```

**依赖说明：**
- Python >= 3.10
- PyTorch >= 2.0
- `bitsandbytes` — 可选，训练时启用 8-bit 优化器
- `matplotlib` — 可选，生成因果热力图时需要
- `datasets` — 可选，使用 HuggingFace 数据源时需要

---

## 快速开始

### 生成

```bash
# 基本生成
python blade_run.py --prompt "Solve: 2x + 5 = 13. Show your work.\n"

# 生成 + 因果图分析
python blade_run.py \
    --prompt "Solve: 2x + 5 = 13. Show your work.\n" \
    --causal \
    --causal_out causal.png
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prompt` | — | 输入提示词 |
| `--beta` | 0.6 | AR/Diff 混合比例（推荐范围 0.49–0.90） |
| `--max_tokens` | 200 | 最大生成 token 数 |
| `--causal` | false | 生成后运行因果图分析 |
| `--causal_out` | causal.png | 因果热力图输出路径 |
| `--save_diagnostics` | — | 将完整诊断信息保存为 JSON |

所有默认值均在 `blade_config.yaml` 中配置。

---

## 配置

编辑 `blade_config.yaml` 来配置模型、数据、训练和推理参数：

```yaml
model:
  base_model: "./base_model"    # 基座模型路径（如 Qwen2.5-0.5B）
  output_dir: "./BLADE-v1"      # 训练后模型保存路径

generate:
  beta:       0.6               # AR/Diff 混合比例
  max_tokens: 200

data:
  sources:
    - name_tag:   "my_data"
      path:       "./data/train.jsonl"
      text_field: "problem,solution"   # 拼接两个字段
      weight:     1
```

### 数据格式

本地 JSONL 文件，每行一个 JSON 对象：
```json
{"problem": "Solve 2x + 5 = 13", "solution": "x = 4"}
{"text": "普通文本字段也支持"}
```

同时支持直接使用 HuggingFace 数据集（通过 `hf_name` 配置）。  
多个数据源可以自由混合并设置不同的采样权重。

---

## 训练

```bash
# 使用默认配置训练
python blade_train.py

# 从 checkpoint 继续训练
python blade_train.py --resume ./ckpt/step20000.pt

# 使用自定义配置
python blade_train.py --config my_config.yaml
```

训练损失组成：

| 损失项 | 权重 | 说明 |
|---|---|---|
| `l_diff` | 1.0 | think 段扩散损失 |
| `l_ar` | `alpha` | clean 段自回归损失 |
| `L_intra` | `lambda_intra` | 块内 token 各异性损失 |

---

## 因果分析

对已保存的诊断文件运行因果分析：

```bash
python blade_causal.py --json diag.json --out causal.png --min_gap 3 --topk 15
```

热力图展示右上象限的 N×N 块级 attention 矩阵。位置 (i, j) 处的非零值表示 clean block i 对 think block j（j < i）存在因果依赖。

---

## 引用

```bibtex
@misc{blade2025,
  title  = {BLADE-LM: Block-Level Autoregressive Diffusion with Causal Discovery},
  author = {Xia Ziye},
  year   = {2025},
  url    = {https://github.com/xiaziye/BLADE-LM}
}
```
