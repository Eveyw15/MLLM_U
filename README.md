# Multimodal Privacy Unlearning — Cross-Modal Leakage Analysis

Honours research on **cross-modal residual leakage** in MLLMs.
Shows that image-conditioned inputs can recover private knowledge even after text-side unlearning, and that lightweight inference-time channel suppression can reduce this leakage.

## Setup (Colab)

```bash
# 1. Clone repos
!git clone https://github.com/your-username/unlearning.git /content/my_unlearning
!git clone https://github.com/Vaidehi99/UnLOK-VQA /content/UnLOK-VQA

# 2. Install dependencies (GPU + 8-bit quantization)
!pip install -r /content/my_unlearning/requirements-colab.txt
```

## Run order

### Step 1 — Download COCO images for the evaluation subset
```bash
!python /content/my_unlearning/download_coco_for_subset.py \
    --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
    --coco_root /content/UnLOK-VQA/data/coco2017 \
    --n 50
```

### Step 2 — Baseline evaluation (dummy image vs. true image)
```bash
!python /content/my_unlearning/minimal_eval_unlok.py \
    --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
    --coco_root /content/UnLOK-VQA/data/coco2017 \
    --result_path /content/UnLOK-VQA/results/minimal_eval_hf_50.jsonl \
    --n 50
```
Output: leakage gap (`true_rate − dummy_rate`). Baseline: gap = 0.34 on N=50.

### Step 3 — Channel sensitivity analysis
```bash
!python /content/my_unlearning/analyze_channels.py \
    --baseline_jsonl /content/UnLOK-VQA/results/minimal_eval_hf_50.jsonl \
    --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
    --coco_root /content/UnLOK-VQA/data/coco2017 \
    --output_json /content/UnLOK-VQA/results/channel_sensitivity.json \
    --layers 28 31 \
    --topk 20
```
Output: top-20 sensitive channels per layer (used by suppression step).

### Step 4 — Evaluation with suppression
```bash
!python /content/my_unlearning/eval_with_suppression.py \
    --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
    --coco_root /content/UnLOK-VQA/data/coco2017 \
    --baseline_jsonl /content/UnLOK-VQA/results/minimal_eval_hf_50.jsonl \
    --sensitivity_json /content/UnLOK-VQA/results/channel_sensitivity.json \
    --result_path /content/UnLOK-VQA/results/suppression_eval.jsonl \
    --layer 31 --topk 20 --alpha 0.0
```

## Key metric

**Leakage gap** = `true_rate − dummy_rate`
A positive gap means the true image enables the model to answer correctly on cases where the dummy image fails — direct evidence of cross-modal leakage.

Goal: suppression reduces the gap while keeping `dummy_rate` (utility proxy) stable.
