"""
eval_with_suppression.py

Three-way evaluation (dummy / true / suppressed) for cross-modal leakage.

Dummy and true results are loaded from the baseline JSONL (already computed by
minimal_eval_unlok.py) to avoid re-running them. Only the suppressed forward
pass is run here.

Single run:
    python eval_with_suppression.py \
        --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
        --coco_root /content/UnLOK-VQA/data/coco2017 \
        --baseline_jsonl /content/UnLOK-VQA/results/minimal_eval_hf_50.jsonl \
        --sensitivity_json /content/UnLOK-VQA/results/channel_sensitivity.json \
        --result_path /content/UnLOK-VQA/results/suppression_eval.jsonl \
        --layer 31 --topk 5 --alpha 0.0

Sanity check (alpha=1.0 should reproduce true image hit rate):
    ... same as above but --alpha 1.0

Sweep (layer 31, topk x alpha grid):
    python eval_with_suppression.py \
        --data_path ... --coco_root ... \
        --baseline_jsonl ... --sensitivity_json ... \
        --result_path /content/UnLOK-VQA/results/suppression_eval.jsonl \
        --layer 31 --sweep

Output (single run):
    <result_path>         — per-sample JSONL
    <result_path>.csv     — one-row summary (all samples + leak-like subset)

Output (sweep):
    <prefix>_sweep_all.csv   — one row per (topk, alpha) combo, full set
    <prefix>_sweep_leak.csv  — same, leak-like subset only

Per-sample JSONL fields:
    id, question, target, image_id, image_found,
    dummy_answer, dummy_match_soft,       <- loaded from baseline JSONL
    true_answer,  true_match_soft,        <- loaded from baseline JSONL
    suppressed_answer, suppressed_match_soft,
    error
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration

from minimal_eval_unlok import (
    find_image,
    clean_answer,
    soft_match,
    load_processor,
    run_with_image,
)
from suppression import ChannelSuppressor, load_top_channels


SWEEP_TOPK   = [1, 3, 5, 10]
SWEEP_ALPHAS = [0.0, 0.2, 0.5, 1.0]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Suppression evaluation: dummy / true / true+suppression"
    )
    p.add_argument("--data_path", required=True,
                   help="Path to zsre_mend_eval.json")
    p.add_argument("--coco_root", required=True,
                   help="Root of COCO images (train2017/ and val2017/)")
    p.add_argument("--baseline_jsonl", required=True,
                   help="Output of minimal_eval_unlok.py (dummy/true results)")
    p.add_argument("--sensitivity_json", required=True,
                   help="channel_sensitivity.json from analyze_channels.py")
    p.add_argument("--result_path", default="results/suppression_eval.jsonl",
                   help="Output JSONL path (single run) or path prefix (sweep)")
    p.add_argument("--layer", type=int, default=31,
                   help="Layer index to suppress (default: 31)")
    p.add_argument("--topk", type=int, default=5,
                   help="Number of top channels to suppress, single-run only (default: 5)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Channel scaling factor: 0.0=zero-out, 1.0=no-op (default: 0.0)")
    p.add_argument("--sweep", action="store_true",
                   help=f"Run topk={SWEEP_TOPK} x alpha={SWEEP_ALPHAS} grid on --layer")
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--no_quantize", action="store_true",
                   help="Disable 8-bit quantization")
    return p.parse_args()


def load_baseline(baseline_jsonl: Path) -> dict[str, dict]:
    """Load per-sample baseline results keyed by sample id."""
    by_id = {}
    with open(baseline_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["id"]] = rec
    return by_id


def load_model_and_processor(args):
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
    return model, processor


def compute_summary(rows: list[dict], label: str = "") -> dict:
    """Aggregate metrics for a list of per-sample result dicts."""
    def rate(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    dummy_rate      = rate("dummy_match_soft")
    true_rate       = rate("true_match_soft")
    suppressed_rate = rate("suppressed_match_soft")

    # Flip counts — only where both true and suppressed results are available
    paired = [r for r in rows
              if r.get("true_match_soft") is not None
              and r.get("suppressed_match_soft") is not None]
    good_flips = sum(1 for r in paired
                     if r["true_match_soft"] and not r["suppressed_match_soft"])
    bad_flips  = sum(1 for r in paired
                     if not r["true_match_soft"] and r["suppressed_match_soft"])

    return {
        "subset":           label,
        "n":                len(rows),
        "dummy_rate":       round(dummy_rate,      4),
        "true_rate":        round(true_rate,       4),
        "suppressed_rate":  round(suppressed_rate, 4),
        "gap_before":       round(true_rate - dummy_rate,       4),  # true − dummy
        "gap_after":        round(suppressed_rate - dummy_rate, 4),  # suppressed − dummy
        "delta":            round(suppressed_rate - true_rate,  4),  # suppressed − true
        "good_flips":       good_flips,   # true=T → suppressed=F  (leakage reduced)
        "bad_flips":        bad_flips,    # true=F → suppressed=T  (noise introduced)
    }


def print_summary(s: dict, layer: int, topk: int, alpha: float):
    print(f"\n===== {s['subset']} (n={s['n']}) =====")
    print(f"  Layer={layer}  top-k={topk}  alpha={alpha}")
    print(f"  Dummy  hit rate       : {s['dummy_rate']:.3f}")
    print(f"  True   hit rate       : {s['true_rate']:.3f}")
    print(f"  Suppressed hit rate   : {s['suppressed_rate']:.3f}")
    print(f"  Leakage gap (before)  : {s['gap_before']:+.3f}  [true − dummy]")
    print(f"  Leakage gap (after)   : {s['gap_after']:+.3f}  [suppressed − dummy]")
    print(f"  Suppression delta     : {s['delta']:+.3f}  [suppressed − true]")
    print(f"  Good flips (T→F)      : {s['good_flips']}")
    print(f"  Bad  flips (F→T)      : {s['bad_flips']}")


def write_csv(csv_path: Path, rows: list[dict]):
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path.resolve()}")


def is_leak_like(r: dict) -> bool:
    return (
        r.get("image_found") is True
        and r.get("dummy_match_soft") is False
        and r.get("true_match_soft") is True
    )


def run_suppressed(model, processor, ex: dict, img: Image.Image,
                   layer: int, channels: list[int], alpha: float,
                   device: str) -> tuple[str | None, bool | None, str | None]:
    """
    Run suppressed forward pass for one sample.
    Returns (suppressed_answer, suppressed_match_soft, error_str).
    """
    try:
        with ChannelSuppressor(model, layer=layer, channels=channels, alpha=alpha):
            raw = run_with_image(model, processor, ex["src"], img, device)
        ans = clean_answer(raw)
        match = soft_match(ex.get("pred", ""), ans)
        return ans, match, None
    except Exception as e:
        return None, None, str(e)


# ── single run ────────────────────────────────────────────────────────────────

def run_single(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path        = Path(args.data_path)
    coco_root        = Path(args.coco_root)
    baseline_jsonl   = Path(args.baseline_jsonl)
    sensitivity_json = Path(args.sensitivity_json)
    result_path      = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    assert data_path.exists(),        f"Not found: {data_path}"
    assert baseline_jsonl.exists(),   f"Not found: {baseline_jsonl}"
    assert sensitivity_json.exists(), f"Not found: {sensitivity_json}"

    channels = load_top_channels(sensitivity_json, layer=args.layer, topk=args.topk)
    print(f"Layer {args.layer}  top-k={args.topk}  alpha={args.alpha}")
    print(f"  Channels: {channels}")

    baseline = load_baseline(baseline_jsonl)
    with open(data_path) as f:
        full_data = json.load(f)
    data = [ex for ex in full_data if ex["id"] in baseline]
    print(f"Eval samples: {len(data)}")

    model, processor = load_model_and_processor(args)

    rows = []
    with open(result_path, "w") as fout:
        for ex in tqdm(data, desc="Evaluating"):
            base     = baseline[ex["id"]]
            img_path = find_image(coco_root, ex["image_id"])

            rec = {
                "id":                  ex["id"],
                "question":            ex["src"],
                "target":              ex.get("pred", ""),
                "image_id":            ex["image_id"],
                "image_found":         img_path is not None,
                # loaded from baseline — not re-run
                "dummy_answer":        base.get("dummy_answer"),
                "dummy_match_soft":    base.get("dummy_match_soft"),
                "true_answer":         base.get("true_answer"),
                "true_match_soft":     base.get("true_match_soft"),
                "suppressed_answer":   None,
                "suppressed_match_soft": None,
                "error":               None,
            }

            if img_path is not None:
                img = Image.open(img_path).convert("RGB")
                ans, match, err = run_suppressed(
                    model, processor, ex, img, args.layer, channels, args.alpha, device
                )
                rec["suppressed_answer"]       = ans
                rec["suppressed_match_soft"]   = match
                rec["error"]                   = err
                if err:
                    print(f"  Error on id={ex['id']}: {err}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    print(f"\nSaved: {result_path.resolve()}")

    # Summaries
    s_all  = compute_summary(rows, label="ALL SAMPLES")
    s_leak = compute_summary([r for r in rows if is_leak_like(r)],
                              label="LEAK-LIKE SUBSET")
    print_summary(s_all,  args.layer, args.topk, args.alpha)
    print_summary(s_leak, args.layer, args.topk, args.alpha)

    csv_path = result_path.with_suffix(".csv")
    write_csv(csv_path, [
        {"layer": args.layer, "topk": args.topk, "alpha": args.alpha, **s_all},
        {"layer": args.layer, "topk": args.topk, "alpha": args.alpha, **s_leak},
    ])


# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path        = Path(args.data_path)
    coco_root        = Path(args.coco_root)
    baseline_jsonl   = Path(args.baseline_jsonl)
    sensitivity_json = Path(args.sensitivity_json)

    result_dir    = Path(args.result_path).parent
    sweep_prefix  = Path(args.result_path).stem
    result_dir.mkdir(parents=True, exist_ok=True)

    assert data_path.exists(),        f"Not found: {data_path}"
    assert baseline_jsonl.exists(),   f"Not found: {baseline_jsonl}"
    assert sensitivity_json.exists(), f"Not found: {sensitivity_json}"

    baseline = load_baseline(baseline_jsonl)
    with open(data_path) as f:
        full_data = json.load(f)
    data = [ex for ex in full_data if ex["id"] in baseline]
    print(f"Sweep: {len(SWEEP_TOPK)} topk × {len(SWEEP_ALPHAS)} alpha = "
          f"{len(SWEEP_TOPK)*len(SWEEP_ALPHAS)} combos  |  {len(data)} samples")

    # Load all top channels up front (we only need the max topk slice)
    max_topk = max(SWEEP_TOPK)
    all_channels = load_top_channels(sensitivity_json, layer=args.layer, topk=max_topk)
    channel_sets = {k: all_channels[:k] for k in SWEEP_TOPK}

    # Pre-load true images to avoid repeated disk I/O across combos
    imgs: dict[str, Image.Image | None] = {}
    for ex in data:
        img_path = find_image(coco_root, ex["image_id"])
        imgs[ex["id"]] = Image.open(img_path).convert("RGB") if img_path else None

    model, processor = load_model_and_processor(args)

    summary_all:  list[dict] = []
    summary_leak: list[dict] = []

    combos = [(k, a) for k in SWEEP_TOPK for a in SWEEP_ALPHAS]
    for topk, alpha in tqdm(combos, desc="Sweep"):
        channels = channel_sets[topk]
        rows = []

        for ex in data:
            base = baseline[ex["id"]]
            img  = imgs[ex["id"]]

            rec = {
                "id":                    ex["id"],
                "image_found":           img is not None,
                "dummy_match_soft":      base.get("dummy_match_soft"),
                "true_match_soft":       base.get("true_match_soft"),
                "suppressed_match_soft": None,
            }

            if img is not None:
                _, match, err = run_suppressed(
                    model, processor, ex, img, args.layer, channels, alpha, device
                )
                rec["suppressed_match_soft"] = match
                if err:
                    print(f"  Error topk={topk} alpha={alpha} id={ex['id']}: {err}")

            rows.append(rec)

        s_all  = compute_summary(rows, label="all")
        s_leak = compute_summary([r for r in rows if is_leak_like(r)],
                                  label="leak_like")

        summary_all.append( {"layer": args.layer, "topk": topk, "alpha": alpha, **s_all})
        summary_leak.append({"layer": args.layer, "topk": topk, "alpha": alpha, **s_leak})

        # Inline progress line
        print(f"  topk={topk:2d}  alpha={alpha:.1f}  "
              f"supp={s_all['suppressed_rate']:.3f}  delta={s_all['delta']:+.4f}  "
              f"leak_supp={s_leak['suppressed_rate']:.3f}  "
              f"leak_delta={s_leak['delta']:+.4f}  "
              f"good={s_leak['good_flips']}  bad={s_leak['bad_flips']}")

    # Write CSVs
    write_csv(result_dir / f"{sweep_prefix}_sweep_all.csv",  summary_all)
    write_csv(result_dir / f"{sweep_prefix}_sweep_leak.csv", summary_leak)

    # Compact printed tables
    header = (f"{'topk':>6}  {'alpha':>5}  {'dummy':>6}  {'true':>6}  "
              f"{'supp':>6}  {'delta':>7}  {'good':>5}  {'bad':>4}")
    row_fmt = (lambda r:
               f"{r['topk']:>6}  {r['alpha']:>5.1f}  {r['dummy_rate']:>6.3f}  "
               f"{r['true_rate']:>6.3f}  {r['suppressed_rate']:>6.3f}  "
               f"{r['delta']:>+7.4f}  {r['good_flips']:>5}  {r['bad_flips']:>4}")

    print(f"\n===== SWEEP — ALL SAMPLES (layer={args.layer}) =====")
    print(header)
    for r in summary_all:
        print(row_fmt(r))

    print(f"\n===== SWEEP — LEAK-LIKE SUBSET (layer={args.layer}) =====")
    print(header)
    for r in summary_leak:
        print(row_fmt(r))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
