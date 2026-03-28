"""
minimal_eval_unlok.py

Baseline evaluation on UnLOK-VQA subset.
Compares dummy (blank) image vs true COCO image using LLaVA-1.5-7B.

Usage:
    python minimal_eval_unlok.py \
        --data_path data/zsre_mend_eval.json \
        --coco_root /path/to/coco2017 \
        --result_path results/minimal_eval_hf_50.jsonl \
        --n 50 \
        --model_id llava-hf/llava-1.5-7b-hf

Requires: transformers, torch, pillow, tqdm, pandas
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
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


def clean_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1].strip()
    return text


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    for ch in list('.,!?:;"\'()[]{}'):
        s = s.replace(ch, " ")
    return " ".join(s.split())


# Simple alias table — extend as needed
_ALIASES: dict[str, set[str]] = {
    "lab": {"lab", "labrador", "labrador retriever", "black lab", "black labrador"},
    "mountain": {"mountain", "mountains", "mountainous"},
    "mountains": {"mountain", "mountains", "mountainous"},
    "police": {"police", "police station", "cop", "officer"},
    "fire": {"fire", "flame", "burning"},
    "country": {"country", "countryside", "rural area"},
}


def get_aliases(target: str) -> set[str]:
    t = normalize_text(target)
    aliases = {t}
    if t in _ALIASES:
        aliases |= _ALIASES[t]
    aliases.add(t[:-1] if t.endswith("s") else t + "s")
    return aliases


def soft_match(target: str, answer: str) -> bool:
    ans = normalize_text(answer)
    return any(alias in ans for alias in get_aliases(target))


# ── model I/O ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_with_image(model, processor, question: str, image: Image.Image,
                   device: str, max_new_tokens: int = 64) -> str:
    prompt = build_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True)


def load_processor(model_id: str):
    try:
        return LlavaProcessor.from_pretrained(model_id, backend="pil")
    except TypeError:
        return LlavaProcessor.from_pretrained(model_id, use_fast=False)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline evaluation: dummy vs true image")
    p.add_argument("--data_path", default="data/zsre_mend_eval.json")
    p.add_argument("--coco_root", required=True,
                   help="Root of COCO images (contains train2017/ and val2017/)")
    p.add_argument("--result_path", default="results/minimal_eval_hf_50.jsonl")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--no_quantize", action="store_true",
                   help="Disable 8-bit quantization (use fp16 instead)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path(args.data_path)
    coco_root = Path(args.coco_root)
    result_path = Path(args.result_path)

    assert data_path.exists(), f"Data file not found: {data_path}"
    result_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Load data
    with open(data_path) as f:
        data = json.load(f)
    data = data[:args.n]
    print(f"Loaded {len(data)} samples")

    # Run evaluation
    rows = []
    with open(result_path, "w") as fout:
        for ex in tqdm(data, desc="Evaluating"):
            q = ex["src"]
            iid = ex["image_id"]
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
                "error": None,
            }

            try:
                dummy_raw = run_with_image(model, processor, q, DUMMY_IMAGE, device)
                rec["dummy_answer"] = clean_answer(dummy_raw)
                rec["dummy_match_soft"] = soft_match(target, rec["dummy_answer"])

                if img_path is not None:
                    true_img = Image.open(img_path).convert("RGB")
                    true_raw = run_with_image(model, processor, q, true_img, device)
                    rec["true_answer"] = clean_answer(true_raw)
                    rec["true_match_soft"] = soft_match(target, rec["true_answer"])

            except Exception as e:
                rec["error"] = str(e)
                print(f"  Error on id={ex['id']}: {e}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    print(f"\nSaved to: {result_path.resolve()}")

    # Summary
    df = pd.DataFrame(rows)
    n_total = len(df)
    n_img_found = int(df["image_found"].sum())

    dummy_rate = df["dummy_match_soft"].fillna(False).astype(int).mean()
    true_rate = df["true_match_soft"].fillna(False).astype(int).mean()

    n_errors = int(df["error"].notna().sum())
    leakage_gap = true_rate - dummy_rate

    print("\n===== SUMMARY =====")
    print(f"Total samples       : {n_total}")
    print(f"Images found        : {n_img_found}/{n_total}")
    print(f"Errors              : {n_errors}")
    print(f"Dummy soft hit rate : {dummy_rate:.3f}")
    print(f"True  soft hit rate : {true_rate:.3f}")
    print(f"Leakage gap (Δ)     : {leakage_gap:+.3f}  "
          f"({'true > dummy — image leaks info' if leakage_gap > 0 else 'no positive gap'})")

    print("\n===== FIRST 5 ROWS =====")
    print(df[[
        "id", "target", "image_found",
        "dummy_answer", "dummy_match_soft",
        "true_answer", "true_match_soft",
        "error",
    ]].head().to_string())


if __name__ == "__main__":
    main()
