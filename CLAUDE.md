# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Honours research on **cross-modal residual leakage** in multimodal LLMs (MLLMs).
Research question: even after text-side unlearning, can private knowledge be recovered via image-conditioned inputs?
Method: lightweight **inference-time selective channel suppression** — no retraining.

## Scope is fixed

Do not expand scope. Do not add training pipelines, new datasets, or abstractions beyond what the five scripts require. Three similar lines of code is preferable to a premature abstraction.

## Environment assumptions

- Model: `llava-hf/llava-1.5-7b-hf` (Vicuna-7B backbone, 32 transformer layers)
- Dataset: UnLOK-VQA — `zsre_mend_eval.json`, COCO 2017 images
- Runs locally for editing; heavy GPU runs on **Google Colab**
- All scripts must be runnable as `python script.py --arg value` (no notebook-only logic)
- 8-bit quantization via `BitsAndBytesConfig` is the default; `--no_quantize` flag disables it

## File structure and responsibilities

| File | Role |
|---|---|
| `download_coco_for_subset.py` | Download/verify COCO images for the first N samples |
| `minimal_eval_unlok.py` | Baseline eval: dummy image vs true image; outputs JSONL + leakage gap |
| `analyze_channels.py` | Hook-based hidden-state capture; outputs top-k sensitive channels per layer to JSON |
| `suppression.py` | Inference-time forward hook that scales selected channels by alpha |
| `eval_with_suppression.py` | Compares dummy / true / true+suppression; reports leakage-related metrics |

## Data flow between scripts

```
download_coco_for_subset.py
        ↓ (images on disk)
minimal_eval_unlok.py  →  results/minimal_eval_hf_50.jsonl
        ↓ (leak-like filter: image_found=T, dummy_match=F, true_match=T)
analyze_channels.py    →  results/channel_sensitivity.json
        ↓ (top_channels per layer)
suppression.py         (imported by eval_with_suppression.py)
        ↓
eval_with_suppression.py  →  results/suppression_eval.jsonl + summary
```

## Key conventions

**Leak-like case definition** (used in analyze_channels and eval_with_suppression):
```python
image_found == True and dummy_match_soft == False and true_match_soft == True
```

**Hook target for LLaVA language model layers:**
```python
model.language_model.model.layers[idx]
```
Layer outputs are tuples; index `[0]` is the hidden state `[batch, seq_len, hidden_dim]`.

**Soft match logic:** `normalize_text(target)` aliases are checked against the model answer. The alias table lives in `minimal_eval_unlok.py` — extend it there, not elsewhere.

**Prompt format:**
```
USER: <image>\n{question}\nASSISTANT:
```

## Colab workflow

```bash
# First-time setup in Colab
!git clone https://github.com/your-username/unlearning.git /content/my_unlearning
!git clone https://github.com/Vaidehi99/UnLOK-VQA /content/UnLOK-VQA

# Update after local changes
%cd /content/my_unlearning && !git pull
```

All scripts take absolute paths as arguments so they can be run from any working directory.

## Baseline results (N=50, reference)

- dummy_rate = 0.24, true_rate = 0.58, gap = 0.34, leak-like cases = 22
- This gap is the target leakage signal that suppression should reduce.
