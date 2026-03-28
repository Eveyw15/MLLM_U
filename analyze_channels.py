"""
analyze_channels.py

Channel sensitivity analysis for cross-modal leakage in LLaVA-1.5-7B.

Uses forward hooks on selected transformer layers to capture hidden states
under dummy image vs. true image inputs. Computes channel-wise sensitivity
as the mean absolute difference across tokens, averaged over leak-like samples.

Only processes "leak-like" cases:
    image_found == True AND dummy_match_soft == False AND true_match_soft == True

Usage:
    python analyze_channels.py \
        --baseline_jsonl results/minimal_eval_hf_50.jsonl \
        --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
        --coco_root /content/UnLOK-VQA/data/coco2017 \
        --output_json results/channel_sensitivity.json \
        --layers 28 31 \
        --topk 20 \
        --model_id llava-hf/llava-1.5-7b-hf
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaProcessor


COCO_SPLITS = ["train2017", "val2017"]
DUMMY_IMAGE = Image.new("RGB", (336, 336), color=(255, 255, 255))


# ── helpers ──────────────────────────────────────────────────────────────────

def find_image(coco_root: Path, image_id: int):
    fn = f"{image_id:012d}.jpg"
    for split in COCO_SPLITS:
        p = coco_root / split / fn
        if p.exists():
            return p
    return None


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def load_leak_like_cases(baseline_jsonl: Path, data_path: Path) -> list[dict]:
    """
    Filter baseline JSONL for leak-like cases, then attach the full data record
    (for the question text) from the original JSON.
    """
    # Index full data by id
    with open(data_path) as f:
        full_data = json.load(f)
    by_id = {ex["id"]: ex for ex in full_data}

    leak_cases = []
    with open(baseline_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if (
                rec.get("image_found") is True
                and rec.get("dummy_match_soft") is False
                and rec.get("true_match_soft") is True
            ):
                # Attach original data record for question etc.
                orig = by_id.get(rec["id"])
                if orig is not None:
                    rec["_orig"] = orig
                    leak_cases.append(rec)

    return leak_cases


# ── hook machinery ────────────────────────────────────────────────────────────

class HiddenStateCapture:
    """Captures the output hidden state of a single transformer layer."""

    def __init__(self):
        self.state: torch.Tensor | None = None

    def hook(self, module, input, output):
        # LLaMA layer output is a tuple; first element is the hidden state
        h = output[0]              # shape: [batch, seq_len, hidden_dim]
        self.state = h.detach().cpu().float()

    def clear(self):
        self.state = None


def register_hooks(model, layer_indices: list[int]) -> tuple[dict[int, HiddenStateCapture], list]:
    """Register hooks on the specified language model layers. Returns captures and handle list."""
    captures = {}
    handles = []
    for idx in layer_indices:
        cap = HiddenStateCapture()
        layer = model.language_model.model.layers[idx]
        handle = layer.register_forward_hook(cap.hook)
        captures[idx] = cap
        handles.append(handle)
    return captures, handles


def remove_hooks(handles: list):
    for h in handles:
        h.remove()


# ── inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def forward_capture(model, processor, question: str, image: Image.Image,
                    captures: dict, device: str) -> dict[int, torch.Tensor]:
    """Run a single forward pass and return captured hidden states per layer index."""
    for cap in captures.values():
        cap.clear()

    prompt = build_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # We only need one forward pass (no generation needed for hidden state capture)
    _ = model(**inputs)

    return {idx: cap.state.clone() for idx, cap in captures.items() if cap.state is not None}


# ── analysis ──────────────────────────────────────────────────────────────────

def compute_channel_sensitivity(
    model, processor, leak_cases: list[dict],
    layer_indices: list[int], coco_root: Path, device: str
) -> dict[int, np.ndarray]:
    """
    For each layer, accumulate mean-abs channel differences across all leak-like samples.
    Returns dict: layer_idx -> np.ndarray of shape [hidden_dim], averaged over samples.
    """
    captures, handles = register_hooks(model, layer_indices)

    accum: dict[int, np.ndarray] = {}  # layer_idx -> running sum
    counts: dict[int, int] = {}

    try:
        for rec in tqdm(leak_cases, desc="Analyzing channels"):
            q = rec["_orig"]["src"]
            iid = rec["image_id"]
            img_path = find_image(coco_root, iid)
            if img_path is None:
                print(f"  Skipping {iid}: image not found")
                continue

            true_img = Image.open(img_path).convert("RGB")

            try:
                h_dummy = forward_capture(model, processor, q, DUMMY_IMAGE, captures, device)
                h_true  = forward_capture(model, processor, q, true_img,   captures, device)
            except Exception as e:
                print(f"  Error on id={rec['id']}: {e}")
                continue

            for idx in layer_indices:
                if idx not in h_dummy or idx not in h_true:
                    continue

                hd = h_dummy[idx]  # [1, seq_len, hidden_dim]
                ht = h_true[idx]

                # Align sequence lengths (should be identical, but guard just in case)
                min_seq = min(hd.shape[1], ht.shape[1])
                hd = hd[:, :min_seq, :]
                ht = ht[:, :min_seq, :]

                # Channel-wise sensitivity: mean over tokens of |h_true - h_dummy|
                diff = (ht - hd).abs().mean(dim=1).squeeze(0).numpy()  # [hidden_dim]

                if idx not in accum:
                    accum[idx] = np.zeros_like(diff)
                    counts[idx] = 0
                accum[idx] += diff
                counts[idx] += 1

    finally:
        remove_hooks(handles)

    # Average over samples
    return {idx: accum[idx] / counts[idx] for idx in accum if counts[idx] > 0}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Channel sensitivity analysis for cross-modal leakage")
    p.add_argument("--baseline_jsonl", required=True,
                   help="Path to baseline JSONL output (from minimal_eval_unlok.py)")
    p.add_argument("--data_path", required=True,
                   help="Path to zsre_mend_eval.json")
    p.add_argument("--coco_root", required=True,
                   help="Root of COCO images")
    p.add_argument("--output_json", default="results/channel_sensitivity.json")
    p.add_argument("--layers", type=int, nargs="+", default=[28, 31],
                   help="Layer indices to analyze (default: 28 31)")
    p.add_argument("--topk", type=int, default=20,
                   help="Top-k sensitive channels to report per layer")
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--no_quantize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_jsonl = Path(args.baseline_jsonl)
    data_path = Path(args.data_path)
    coco_root = Path(args.coco_root)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    assert baseline_jsonl.exists(), f"Baseline JSONL not found: {baseline_jsonl}"
    assert data_path.exists(), f"Data file not found: {data_path}"

    # Load leak-like subset
    leak_cases = load_leak_like_cases(baseline_jsonl, data_path)
    print(f"Leak-like cases: {len(leak_cases)}")
    if len(leak_cases) == 0:
        print("No leak-like cases found. Check baseline JSONL filters.")
        return 1

    # Load model
    quant_config = None if args.no_quantize else BitsAndBytesConfig(load_in_8bit=True)
    print(f"Loading processor: {args.model_id}")
    processor = LlavaProcessor.from_pretrained(args.model_id, use_fast=False)
    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()

    n_layers = len(model.language_model.model.layers)
    print(f"LM has {n_layers} layers. Analyzing: {args.layers}")
    for idx in args.layers:
        assert 0 <= idx < n_layers, f"Layer {idx} out of range [0, {n_layers-1}]"

    # Run analysis
    sensitivity = compute_channel_sensitivity(
        model, processor, leak_cases, args.layers, coco_root, device
    )

    # Build top-k results
    results = {}
    for idx, scores in sensitivity.items():
        top_indices = np.argsort(scores)[::-1][:args.topk].tolist()
        top_scores  = scores[top_indices].tolist()
        results[str(idx)] = {
            "top_channels": top_indices,
            "scores": [round(s, 6) for s in top_scores],
        }
        print(f"\nLayer {idx} — top {args.topk} channels:")
        print(f"  indices: {top_indices[:10]} ...")
        print(f"  scores:  {[round(s, 4) for s in top_scores[:10]]} ...")

    out = {
        "layers": args.layers,
        "n_samples": len(leak_cases),
        "k": args.topk,
        "results": results,
    }

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {output_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
