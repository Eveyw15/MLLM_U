"""
eval_with_suppression.py

Three-way evaluation for cross-modal leakage with inference-time suppression.

Compares:
  1. dummy image            — no-information baseline
  2. true image             — leakage upper bound
  3. true image + suppression — target condition

Reports leakage gap before and after suppression.

Usage:
    python eval_with_suppression.py \
        --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
        --coco_root /content/UnLOK-VQA/data/coco2017 \
        --baseline_jsonl /content/UnLOK-VQA/results/minimal_eval_hf_50.jsonl \
        --sensitivity_json /content/UnLOK-VQA/results/channel_sensitivity.json \
        --result_path /content/UnLOK-VQA/results/suppression_eval.jsonl \
        --layer 31 --topk 5 --alpha 0.0

Output JSONL fields per record:
    id, question, target, image_id, image_found,
    dummy_answer, dummy_match_soft,
    true_answer, true_match_soft,
    suppressed_answer, suppressed_match_soft,
    error
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration

from minimal_eval_unlok import (
    find_image,
    build_prompt,
    clean_answer,
    soft_match,
    load_processor,
    run_with_image,
    DUMMY_IMAGE,
    COCO_SPLITS,
)
from suppression import ChannelSuppressor, load_top_channels


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-way evaluation: dummy / true / true+suppression"
    )
    p.add_argument("--data_path", required=True,
                   help="Path to zsre_mend_eval.json")
    p.add_argument("--coco_root", required=True,
                   help="Root of COCO images (contains train2017/ and val2017/)")
    p.add_argument("--baseline_jsonl", required=True,
                   help="Baseline JSONL from minimal_eval_unlok.py (used to select eval subset)")
    p.add_argument("--sensitivity_json", required=True,
                   help="channel_sensitivity.json from analyze_channels.py")
    p.add_argument("--result_path", default="results/suppression_eval.jsonl")
    p.add_argument("--layer", type=int, default=31,
                   help="Layer to suppress (default: 31)")
    p.add_argument("--topk", type=int, default=5,
                   help="Number of top channels to suppress (default: 5)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Channel scaling factor: 0.0=zero-out, 1.0=no-op (default: 0.0)")
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--no_quantize", action="store_true",
                   help="Disable 8-bit quantization (use fp16 instead)")
    return p.parse_args()


def load_eval_ids(baseline_jsonl: Path) -> set[str]:
    """Collect the IDs evaluated in the baseline run."""
    ids = set()
    with open(baseline_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            ids.add(rec["id"])
    return ids


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path       = Path(args.data_path)
    coco_root       = Path(args.coco_root)
    baseline_jsonl  = Path(args.baseline_jsonl)
    sensitivity_json = Path(args.sensitivity_json)
    result_path     = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    assert data_path.exists(),        f"Data file not found: {data_path}"
    assert baseline_jsonl.exists(),   f"Baseline JSONL not found: {baseline_jsonl}"
    assert sensitivity_json.exists(), f"Sensitivity JSON not found: {sensitivity_json}"

    # Load channels to suppress
    channels = load_top_channels(sensitivity_json, layer=args.layer, topk=args.topk)
    print(f"Layer {args.layer} — suppressing top-{args.topk} channels (alpha={args.alpha})")
    print(f"  Channel indices: {channels}")

    # Filter data to the baseline subset
    eval_ids = load_eval_ids(baseline_jsonl)
    with open(data_path) as f:
        full_data = json.load(f)
    data = [ex for ex in full_data if ex["id"] in eval_ids]
    print(f"Eval samples: {len(data)}")

    # Load model
    quant_config = None if args.no_quantize else BitsAndBytesConfig(load_in_8bit=True)
    print(f"Loading processor: {args.model_id}")
    processor = load_processor(args.model_id)
    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()

    # Evaluate
    rows = []
    with open(result_path, "w") as fout:
        for ex in tqdm(data, desc="Evaluating"):
            q      = ex["src"]
            iid    = ex["image_id"]
            target = ex.get("pred", "")
            img_path = find_image(coco_root, iid)

            rec = {
                "id": ex["id"],
                "question": q,
                "target": target,
                "image_id": iid,
                "image_found": img_path is not None,
                "dummy_answer": None,
                "dummy_match_soft": None,
                "true_answer": None,
                "true_match_soft": None,
                "suppressed_answer": None,
                "suppressed_match_soft": None,
                "error": None,
            }

            try:
                # Condition 1: dummy image
                rec["dummy_answer"] = clean_answer(
                    run_with_image(model, processor, q, DUMMY_IMAGE, device)
                )
                rec["dummy_match_soft"] = soft_match(target, rec["dummy_answer"])

                if img_path is not None:
                    true_img = Image.open(img_path).convert("RGB")

                    # Condition 2: true image (no suppression)
                    rec["true_answer"] = clean_answer(
                        run_with_image(model, processor, q, true_img, device)
                    )
                    rec["true_match_soft"] = soft_match(target, rec["true_answer"])

                    # Condition 3: true image + suppression
                    with ChannelSuppressor(model, layer=args.layer,
                                          channels=channels, alpha=args.alpha):
                        raw_suppressed = run_with_image(model, processor, q, true_img, device)
                    rec["suppressed_answer"] = clean_answer(raw_suppressed)
                    rec["suppressed_match_soft"] = soft_match(target, rec["suppressed_answer"])

            except Exception as e:
                rec["error"] = str(e)
                print(f"  Error on id={ex['id']}: {e}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    print(f"\nSaved to: {result_path.resolve()}")

    # Summary
    n_total   = len(rows)
    n_img     = sum(1 for r in rows if r["image_found"])
    n_errors  = sum(1 for r in rows if r["error"])

    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    dummy_rate      = rate("dummy_match_soft")
    true_rate       = rate("true_match_soft")
    suppressed_rate = rate("suppressed_match_soft")

    gap_before = true_rate - dummy_rate
    gap_after  = suppressed_rate - dummy_rate
    delta      = suppressed_rate - true_rate  # negative = suppression reduced leakage

    print("\n===== SUPPRESSION RESULTS =====")
    print(f"Total samples          : {n_total}")
    print(f"Images found           : {n_img}/{n_total}")
    print(f"Errors                 : {n_errors}")
    print(f"Dummy soft hit rate    : {dummy_rate:.3f}")
    print(f"True  soft hit rate    : {true_rate:.3f}")
    print(f"Suppressed hit rate    : {suppressed_rate:.3f}")
    print(f"Leakage gap (before)   : {gap_before:+.3f}  [true − dummy]")
    print(f"Leakage gap (after)    : {gap_after:+.3f}  [suppressed − dummy]")
    print(f"Suppression delta      : {delta:+.3f}  [suppressed − true]")
    print(f"\nLayer={args.layer}  top-k={args.topk}  alpha={args.alpha}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
