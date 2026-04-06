# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Honours research on **cross-modal residual leakage** in multimodal LLMs. The hypothesis: even after text-side unlearning, private knowledge can be recovered via image-conditioned prompts. The fix: lightweight inference-time selective channel suppression (no retraining). Model: LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`). Dataset: UnLOK-VQA (`zsre_mend_eval.json`) with COCO 2017 images.

Dont't use '—' when generate context.

## Environment

All heavy scripts (steps 2–4) require a GPU and 8-bit quantization. They are designed to run on **Google Colab**. Steps 1 and the smoke-test in `suppression.py` run locally without GPU. Install dependencies:

```bash
# Colab (GPU, quantization)
pip install -r requirements-colab.txt

# Local (no GPU needed, for download + smoke-tests only)
pip install -r requirements.txt
```

## Five-script pipeline


| Script                        | Inputs                                            | Outputs                     |
| ----------------------------- | ------------------------------------------------- | --------------------------- |
| `download_coco_for_subset.py` | `zsre_mend_eval.json`                             | COCO images on disk         |
| `minimal_eval_unlok.py`       | `zsre_mend_eval.json`, COCO images                | `minimal_eval_hf_50.jsonl`  |
| `analyze_channels.py`         | baseline JSONL,`zsre_mend_eval.json`, COCO images | `channel_sensitivity.json`  |
| `suppression.py` (smoke-test) | `channel_sensitivity.json`                        | prints validation, no files |
| `eval_with_suppression.py`    | all of the above                                  | `suppression_eval.jsonl`    |

Each script is fully standalone with `argparse`. See `README.md` for exact CLI commands.

## Key data flow

- **Leak-like cases** (used by `analyze_channels.py`): records in baseline JSONL where `image_found=True AND dummy_match_soft=False AND true_match_soft=True`. These are the cases where the true image enables correct answering but the dummy image does not — direct evidence of leakage.
- **Dummy image**: `Image.new("RGB", (336, 336), color=(255, 255, 255))` — blank white, used as the no-information baseline.
- **Prompt format** (must match LLaVA chat template): `"USER: <image>\n{question}\nASSISTANT:"`
- **Soft match** (`soft_match()` in `minimal_eval_unlok.py`): normalized alias-based substring match. Import from `minimal_eval_unlok` in downstream scripts.
- **Layer path resolution** (`get_language_layers()` in `analyze_channels.py`): handles the two attribute paths that differ across `transformers` versions:
  - Older: `model.language_model.model.layers`
  - Newer: `model.model.language_model.layers`
  - `suppression.py` imports this function from `analyze_channels` to avoid duplication.

## Channel sensitivity JSON schema

```json
{
  "layers": [28, 31],
  "n_samples": 18,
  "k": 20,
  "results": {
    "31": {"top_channels": [1512, ...], "scores": [30.30, ...]},
    "28": {"top_channels": [2048, ...], "scores": [15.07, ...]}
  }
}
```

## Suppressor API

```python
from suppression import ChannelSuppressor, load_top_channels
channels = load_top_channels("results/channel_sensitivity.json", layer=31, topk=20)
with ChannelSuppressor(model, layer=31, channels=channels, alpha=0.0):
    output = model.generate(...)
```

`alpha=0.0` zeros the channels; `alpha=1.0` is a no-op. The hook applies `h[:, :, channels] *= alpha` to the layer output tensor.

## Key metric

**Leakage gap** = `true_rate − dummy_rate`. Baseline (N=50): gap ≈ +0.34. Goal: suppression reduces gap while `dummy_rate` stays stable (utility proxy).

## Colab workflow

The repo is cloned into Colab at `/content/my_unlearning`. UnLOK-VQA data lives at `/content/UnLOK-VQA/`. Results write to `/content/UnLOK-VQA/results/`. All CLI commands in `README.md` use these paths.

## LlavaProcessor compatibility

`LlavaProcessor.from_pretrained()` API changed across `transformers` versions. Always use the `load_processor()` helper (defined in both `minimal_eval_unlok.py` and `analyze_channels.py`):

```python
def load_processor(model_id: str):
    try:
        return LlavaProcessor.from_pretrained(model_id, backend="pil")
    except TypeError:
        return LlavaProcessor.from_pretrained(model_id, use_fast=False)
```
