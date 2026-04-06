"""
gating_module.py

Input-conditioned channel gate for cross-modal leakage suppression.

Unlike the fixed suppression in suppression.py (same alpha for all inputs),
this module predicts a per-sample gate vector from the hidden activations
of top-k sensitive channels. The gate can learn to suppress leakage on
forget-like inputs while staying open on retain-like inputs.

Architecture (layer 31, k = number of sensitive channels):

    h_k = h[:, :, top_k_indices]              # [B, T, k]
    x   = pool_fn(h_k)                        # [B, k]  (mean of abs values)
    g   = sigmoid(W2 * ReLU(W1 * x + b1) + b2)  # [B, k]
    h[:, :, top_k_indices] *= g.unsqueeze(1)   # broadcast over T

Training loss:
    L = L_forget + lambda_retain * L_retain
    L_forget = -cross_entropy(target | true_image, gated_model)
    L_retain =  cross_entropy(original_output | image, gated_model)

Usage (training):
    python gating_module.py \
        --data_path /content/UnLOK-VQA/data/zsre_mend_eval.json \
        --coco_root /content/UnLOK-VQA/data/coco2017 \
        --baseline_jsonl /content/UnLOK-VQA/results/minimal_eval_hf_50.jsonl \
        --sensitivity_json /content/UnLOK-VQA/results/channel_sensitivity.json \
        --output_dir /content/UnLOK-VQA/results/gate_v2 \
        --layer 31 --topk 5 --epochs 5 --lr 0.01

Usage (importable):
    from gating_module import InputConditionedGate, GateHook, GateTrainer
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =========================================================================
#  1. Pooling (swappable)
# =========================================================================

def pool_abs_mean(h_k: torch.Tensor) -> torch.Tensor:
    """
    Pool over the token dimension using mean of absolute values.

    Args:
        h_k: [B, T, k] hidden states for selected channels
    Returns:
        [B, k] pooled activation statistics
    """
    return h_k.abs().mean(dim=1)


# =========================================================================
#  2. Input-conditioned gate module
# =========================================================================

class InputConditionedGate(nn.Module):
    """
    Lightweight MLP that predicts a per-sample gate vector from pooled
    activation statistics of the top-k sensitive channels.

    Input:  pooled abs activations [B, k]
    Output: gate values in [0, 1]  [B, k]

    Conservative initialization: final bias set so that sigmoid output
    starts near 0.85-0.9 (gates mostly open). This avoids breaking the
    model before any training.
    """

    def __init__(self, k: int, final_bias_init: float = 2.0,
                 pool_fn=pool_abs_mean):
        """
        Args:
            k: number of sensitive channels (gate input/output width)
            final_bias_init: initial value for the output bias.
                sigmoid(2.0) ~ 0.88, sigmoid(2.5) ~ 0.92.
            pool_fn: callable [B, T, k] -> [B, k]
        """
        super().__init__()
        self.k = k
        self.pool_fn = pool_fn

        self.fc1 = nn.Linear(k, k)
        self.fc2 = nn.Linear(k, k)

        # Conservative init: small weights, bias set for mostly-open output
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.constant_(self.fc2.bias, final_bias_init)

    def forward(self, h_k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_k: [B, T, k] hidden states for top-k channels
        Returns:
            gate: [B, k] values in [0, 1]
        """
        x = self.pool_fn(h_k)       # [B, k]
        x = F.relu(self.fc1(x))     # [B, k]
        x = self.fc2(x)             # [B, k]
        return torch.sigmoid(x)     # [B, k]

    @torch.no_grad()
    def get_gate_values(self, h_k: torch.Tensor) -> torch.Tensor:
        """Detached gate values for inspection."""
        return self.forward(h_k).cpu()


# =========================================================================
#  3. Gate hook (context manager)
# =========================================================================

class GateHook:
    """
    Applies an InputConditionedGate to a transformer layer via forward hook.

    The hook:
      1. Extracts h[:, :, channel_indices] from the layer output
      2. Passes it through the gate MLP
      3. Multiplies the selected channels by the gate output

    Works as context manager or persistent mode:
        with GateHook(model, gate, layer=31, channels=[...]):
            output = model.generate(...)
    """

    def __init__(self, model, gate: InputConditionedGate,
                 layer: int, channel_indices: list[int]):
        self.model = model
        self.gate = gate
        self.layer_idx = layer
        self.channel_indices = channel_indices
        self._handle = None

    def _make_hook(self):
        gate = self.gate
        ch_idx = self.channel_indices

        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                h = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                h = output
                rest = ()
                is_tuple = False

            # Extract top-k channels: [B, T, k]
            h_k = h[:, :, ch_idx]

            # Predict per-sample gate: [B, k]
            g = gate(h_k)

            # Apply gate (broadcast over token dim)
            h = h.clone()
            h[:, :, ch_idx] = h_k * g.unsqueeze(1)  # [B, T, k] * [B, 1, k]

            if is_tuple:
                return (h,) + rest
            return h

        return hook_fn

    def register(self):
        from analyze_channels import get_language_layers
        layers = get_language_layers(self.model)
        self._handle = layers[self.layer_idx].register_forward_hook(self._make_hook())

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


# =========================================================================
#  4. Data split: forget set vs retain set
# =========================================================================

def split_forget_retain(baseline_jsonl: Path, data_path: Path, coco_root: Path):
    """
    Split evaluation samples into forget and retain sets.

    Forget: leak-like cases (dummy=F, true=T, image_found=T)
    Retain: high-quality non-leak cases (image_found=T, true=T, not leak-like)

    Returns two lists of dicts with keys:
        id, question, target, image_path, image_id, teacher_answer
    """
    with open(data_path) as f:
        full_data = json.load(f)
    by_id = {ex["id"]: ex for ex in full_data}
    from minimal_eval_unlok import find_image

    baseline = {}
    with open(baseline_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            baseline[rec["id"]] = rec

    forget_set = []
    retain_set = []

    for sid, rec in baseline.items():
        img_path = find_image(coco_root, rec["image_id"])
        if img_path is None:
            continue

        orig = by_id.get(sid)
        if orig is None:
            continue

        entry = {
            "id": sid,
            "question": orig["src"],
            "target": orig.get("pred", ""),
            "image_path": str(img_path),
            "image_id": rec["image_id"],
            "teacher_answer": rec.get("true_answer") or "",
        }

        is_leak = (
            rec.get("image_found") is True
            and rec.get("dummy_match_soft") is False
            and rec.get("true_match_soft") is True
        )

        is_high_quality_retain = (
            rec.get("image_found") is True
            and rec.get("true_match_soft") is True
            and not is_leak
        )

        if is_leak:
            forget_set.append(entry)
        elif is_high_quality_retain:
            retain_set.append(entry)

    return forget_set, retain_set


# =========================================================================
#  5. Loss functions
# =========================================================================

def build_masked_inputs(processor, image, prompt: str, answer: str, device: str):
    """
    Build prompt + answer inputs, masking prompt tokens so the loss is
    computed only on answer tokens.
    """
    full_text = prompt if not answer else prompt + " " + answer

    prompt_inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    )
    full_inputs = processor(
        images=image, text=full_text, return_tensors="pt"
    )
    full_inputs = full_inputs.to(device)

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return full_inputs


def compute_forget_loss(model, processor, sample: dict,
                        device: str) -> torch.Tensor:
    """
    Forget loss: gradient ascent on the target answer tokens only.
    Returns -cross_entropy so that minimizing this loss makes the model
    worse at producing the target answer, without training on prompt tokens.
    """
    from PIL import Image
    from minimal_eval_unlok import build_prompt

    question = sample["question"]
    target = sample["target"]
    image = Image.open(sample["image_path"]).convert("RGB")

    prompt = build_prompt(question)
    inputs = build_masked_inputs(processor, image, prompt, target, device)
    outputs = model(**inputs)

    return -outputs.loss


def compute_retain_loss(model, processor, sample: dict,
                        device: str) -> torch.Tensor:
    """
    Retain loss: standard cross-entropy on the ungated baseline answer
    tokens only. Minimizing this keeps the gated model close to its
    original answer behavior on retain samples.
    """
    from PIL import Image
    from minimal_eval_unlok import build_prompt

    question = sample["question"]
    teacher_answer = sample.get("teacher_answer", "").strip()
    image = Image.open(sample["image_path"]).convert("RGB")

    if not teacher_answer:
        raise ValueError("Missing baseline true_answer for retain sample")

    prompt = build_prompt(question)
    inputs = build_masked_inputs(processor, image, prompt, teacher_answer, device)
    outputs = model(**inputs)

    return outputs.loss


# =========================================================================
#  6. Trainer
# =========================================================================

class GateTrainer:
    """
    Trains the InputConditionedGate on forget/retain data.

    All model parameters are frozen. Only the gate MLP parameters
    (2 * k * k + 2 * k) are trained.
    """

    def __init__(self, model, processor,
                 gate: InputConditionedGate,
                 layer: int,
                 channel_indices: list[int],
                 device: str,
                 lr: float = 0.01,
                 lambda_retain: float = 1.0):
        self.model = model
        self.processor = processor
        self.gate = gate
        self.layer = layer
        self.channel_indices = channel_indices
        self.device = device
        self.lambda_retain = lambda_retain

        # Freeze all model parameters
        for p in model.parameters():
            p.requires_grad = False

        # Only gate MLP parameters are trainable
        self.optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
        self.gate_hook = GateHook(model, gate, layer, channel_indices)

    def train_epoch(self, forget_set: list[dict], retain_set: list[dict],
                    epoch: int) -> dict:
        """Run one epoch. Returns loss stats."""
        self.gate.train()
        self.gate_hook.register()

        total_forget_loss = 0.0
        total_retain_loss = 0.0
        n_forget = 0
        n_retain = 0

        all_samples = (
            [(s, "forget") for s in forget_set] +
            [(s, "retain") for s in retain_set]
        )
        random.shuffle(all_samples)

        pbar = tqdm(all_samples, desc=f"Epoch {epoch + 1}")
        for sample, split in pbar:
            self.optimizer.zero_grad()

            try:
                if split == "forget":
                    loss = compute_forget_loss(
                        self.model, self.processor, sample, self.device
                    )
                    total_forget_loss += loss.item()
                    n_forget += 1
                else:
                    loss = compute_retain_loss(
                        self.model, self.processor, sample, self.device
                    )
                    loss = loss * self.lambda_retain
                    total_retain_loss += loss.item()
                    n_retain += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gate.parameters(), max_norm=1.0)
                self.optimizer.step()

            except Exception as e:
                print(f"  Error on {sample['id']}: {e}")
                continue

            pbar.set_postfix({
                "f_loss": f"{total_forget_loss / (n_forget or 1):.3f}",
                "r_loss": f"{total_retain_loss / (n_retain or 1):.3f}",
            })

        self.gate_hook.remove()

        return {
            "epoch": epoch + 1,
            "avg_forget_loss": total_forget_loss / max(n_forget, 1),
            "avg_retain_loss": total_retain_loss / max(n_retain, 1),
            "n_forget": n_forget,
            "n_retain": n_retain,
        }

    def evaluate(self, forget_set: list[dict], retain_set: list[dict]) -> dict:
        """Measure gated vs ungated hit rates on forget and retain sets."""
        from PIL import Image
        from minimal_eval_unlok import clean_answer, run_with_image, soft_match

        self.gate.eval()
        results = {"forget": [], "retain": []}

        for split_name, samples in [("forget", forget_set), ("retain", retain_set)]:
            for sample in tqdm(samples, desc=f"Eval {split_name}"):
                img = Image.open(sample["image_path"]).convert("RGB")
                target = sample["target"]

                # Ungated (true image, no gate)
                raw_ungated = run_with_image(
                    self.model, self.processor,
                    sample["question"], img, self.device
                )
                ungated_answer = clean_answer(raw_ungated)
                ungated_match = soft_match(target, ungated_answer)

                # Gated
                with GateHook(self.model, self.gate, self.layer,
                              self.channel_indices):
                    raw_gated = run_with_image(
                        self.model, self.processor,
                        sample["question"], img, self.device
                    )
                gated_answer = clean_answer(raw_gated)
                gated_match = soft_match(target, gated_answer)

                results[split_name].append({
                    "id": sample["id"],
                    "target": target,
                    "ungated_match": ungated_match,
                    "gated_match": gated_match,
                    "ungated_answer": ungated_answer,
                    "gated_answer": gated_answer,
                })

        def rate(records, key):
            vals = [r[key] for r in records]
            return sum(vals) / len(vals) if vals else float("nan")

        forget_ungated = rate(results["forget"], "ungated_match")
        forget_gated = rate(results["forget"], "gated_match")
        retain_ungated = rate(results["retain"], "ungated_match")
        retain_gated = rate(results["retain"], "gated_match")

        forget_good_flips = sum(
            1 for r in results["forget"]
            if r["ungated_match"] and not r["gated_match"]
        )
        forget_bad_flips = sum(
            1 for r in results["forget"]
            if not r["ungated_match"] and r["gated_match"]
        )

        return {
            "forget_ungated_rate": round(forget_ungated, 4),
            "forget_gated_rate": round(forget_gated, 4),
            "forget_good_flips": forget_good_flips,
            "forget_bad_flips": forget_bad_flips,
            "retain_ungated_rate": round(retain_ungated, 4),
            "retain_gated_rate": round(retain_gated, 4),
            "n_forget": len(results["forget"]),
            "n_retain": len(results["retain"]),
            "details": results,
        }

    def save(self, output_dir: Path):
        """Save gate weights, channel indices, and config."""
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.gate.state_dict(),
            "layer": self.layer,
            "channel_indices": self.channel_indices,
            "k": self.gate.k,
        }, output_dir / "gate_weights.pt")
        print(f"Gate saved to: {output_dir / 'gate_weights.pt'}")

    @staticmethod
    def load_gate(path: Path, device: str = "cpu",
                  pool_fn=pool_abs_mean) -> tuple:
        """
        Load a trained gate from disk.
        Returns (gate, layer, channel_indices).
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        gate = InputConditionedGate(k=ckpt["k"], pool_fn=pool_fn)
        gate.load_state_dict(ckpt["state_dict"])
        gate = gate.to(device)
        return gate, ckpt["layer"], ckpt["channel_indices"]


# =========================================================================
#  7. Smoke test (runs locally without GPU)
# =========================================================================

def smoke_test():
    """
    Verify the gate module on synthetic data. No model loading needed.
    Tests: shapes, conservative init, gradient flow, pooling.
    """
    print("=" * 60)
    print("  SMOKE TEST: InputConditionedGate")
    print("=" * 60)

    k = 5
    B, T, H = 2, 10, 4096
    channel_indices = [100, 200, 300, 400, 500]

    gate = InputConditionedGate(k=k, final_bias_init=2.0)

    # ---- 1. Shape check ----
    h_full = torch.randn(B, T, H)
    h_k = h_full[:, :, channel_indices]  # [B, T, k]
    g = gate(h_k)
    assert g.shape == (B, k), f"Expected ({B}, {k}), got {g.shape}"
    print(f"[PASS] Output shape: {g.shape}")

    # ---- 2. Conservative init check ----
    # With small xavier weights and bias=2.0, output should be near sigmoid(2.0) ~ 0.88
    mean_gate = g.mean().item()
    print(f"[INFO] Mean gate value at init: {mean_gate:.4f}")
    assert 0.75 < mean_gate < 0.98, (
        f"Conservative init failed: mean={mean_gate:.4f}, expected ~0.88"
    )
    print(f"[PASS] Conservative init: mean gate {mean_gate:.4f} in [0.75, 0.98]")

    # ---- 3. Gradient flow check ----
    gate.zero_grad()
    loss = g.sum()
    loss.backward()
    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0
                   for p in gate.parameters())
    assert has_grad, "No gradients on gate parameters"
    print("[PASS] Gradients flow to all gate parameters")

    # ---- 4. Pooling check ----
    h_k_test = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])  # [1, 2, 2]
    pooled = pool_abs_mean(h_k_test)
    expected = torch.tensor([[2.0, 3.0]])  # mean(abs) over tokens
    assert torch.allclose(pooled, expected), (
        f"Pooling mismatch: {pooled} vs {expected}"
    )
    print("[PASS] pool_abs_mean correct")

    # ---- 5. Hook simulation ----
    h_before = torch.randn(1, T, H)
    h_k = h_before[:, :, channel_indices]
    g = gate(h_k)  # [1, k]
    h_after = h_before.clone()
    h_after[:, :, channel_indices] = h_k * g.unsqueeze(1)

    # Non-target channels should be unchanged
    mask = torch.ones(H, dtype=torch.bool)
    mask[channel_indices] = False
    diff = (h_after[:, :, mask] - h_before[:, :, mask]).abs().sum().item()
    assert diff == 0.0, f"Non-target channels changed by {diff}"
    print("[PASS] Non-target channels untouched by gate")

    # ---- 6. Parameter count ----
    n_params = sum(p.numel() for p in gate.parameters())
    expected_params = k * k + k + k * k + k  # fc1 weight + bias + fc2 weight + bias
    assert n_params == expected_params, (
        f"Param count: {n_params} vs expected {expected_params}"
    )
    print(f"[PASS] Parameter count: {n_params} (2*{k}*{k} + 2*{k})")

    # ---- 7. Comparison with fixed suppression ----
    # Fixed alpha=0 is equivalent to gate always outputting 0
    print(f"\n[INFO] Fixed suppression (alpha=0): gate = 0 for all inputs")
    print(f"[INFO] Learned gate: can output ~0 for forget, ~1 for retain")
    print(f"[INFO] At init: gate outputs ~{mean_gate:.2f} for all inputs (near identity)")

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60)


# =========================================================================
#  8. CLI entry point
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an input-conditioned channel gate for leakage suppression"
    )
    p.add_argument("--data_path", required=True,
                   help="Path to zsre_mend_eval.json")
    p.add_argument("--coco_root", required=True,
                   help="Root of COCO images (train2017/ and val2017/)")
    p.add_argument("--baseline_jsonl", required=True,
                   help="Output of minimal_eval_unlok.py")
    p.add_argument("--sensitivity_json", required=True,
                   help="channel_sensitivity.json from analyze_channels.py")
    p.add_argument("--output_dir", default="results/gate_v2",
                   help="Directory to save gate weights and logs")
    p.add_argument("--layer", type=int, default=31,
                   help="Target layer (default: 31)")
    p.add_argument("--topk", type=int, default=5,
                   help="Number of top sensitive channels to gate (default: 5)")
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs (default: 5)")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate (default: 0.01)")
    p.add_argument("--lambda_retain", type=float, default=1.0,
                   help="Weight of retain loss (default: 1.0)")
    p.add_argument("--final_bias", type=float, default=2.0,
                   help="Initial bias for gate output layer (default: 2.0, sigmoid~0.88)")
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--no_quantize", action="store_true",
                   help="Disable 8-bit quantization")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run smoke test only (no model loading)")
    return p.parse_args()


def main():
    from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
    from analyze_channels import get_language_layers
    from minimal_eval_unlok import load_processor
    from suppression import load_top_channels

    args = parse_args()

    if args.smoke_test:
        smoke_test()
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path(args.data_path)
    coco_root = Path(args.coco_root)
    baseline_jsonl = Path(args.baseline_jsonl)
    sensitivity_json = Path(args.sensitivity_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert data_path.exists(), f"Not found: {data_path}"
    assert baseline_jsonl.exists(), f"Not found: {baseline_jsonl}"
    assert sensitivity_json.exists(), f"Not found: {sensitivity_json}"

    # -- data split --
    forget_set, retain_set = split_forget_retain(baseline_jsonl, data_path, coco_root)
    print(f"Forget set (leak-like): {len(forget_set)}")
    print(f"Retain set (non-leak):  {len(retain_set)}")

    if len(forget_set) == 0:
        print("No forget samples found. Exiting.")
        return 1

    # -- load sensitive channels --
    channel_indices = load_top_channels(
        sensitivity_json, layer=args.layer, topk=args.topk
    )
    print(f"Layer {args.layer}, top-{args.topk} channels: {channel_indices}")

    # -- load model --
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

    # -- create gate --
    k = len(channel_indices)
    gate = InputConditionedGate(k=k, final_bias_init=args.final_bias)

    # Move gate to same device as the target layer
    layers = get_language_layers(model)
    layer_device = next(layers[args.layer].parameters()).device
    gate = gate.to(layer_device)

    n_params = sum(p.numel() for p in gate.parameters())
    print(f"Gate created: k={k}, {n_params} trainable parameters")

    # -- train --
    trainer = GateTrainer(
        model, processor, gate,
        layer=args.layer,
        channel_indices=channel_indices,
        device=device,
        lr=args.lr,
        lambda_retain=args.lambda_retain,
    )

    log = []
    for epoch in range(args.epochs):
        stats = trainer.train_epoch(forget_set, retain_set, epoch)
        log.append(stats)
        print(f"\n  Epoch {stats['epoch']}: "
              f"forget_loss={stats['avg_forget_loss']:.4f}  "
              f"retain_loss={stats['avg_retain_loss']:.4f}")

    # -- save --
    trainer.save(output_dir)

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log: {output_dir / 'training_log.json'}")

    # -- evaluate --
    print("\n===== POST-TRAINING EVALUATION =====")
    eval_results = trainer.evaluate(forget_set, retain_set)

    print(f"\nForget set ({eval_results['n_forget']} samples):")
    print(f"  Ungated hit rate:  {eval_results['forget_ungated_rate']:.3f}")
    print(f"  Gated hit rate:    {eval_results['forget_gated_rate']:.3f}")
    print(f"  Good flips (T->F): {eval_results['forget_good_flips']}")
    print(f"  Bad flips (F->T):  {eval_results['forget_bad_flips']}")

    print(f"\nRetain set ({eval_results['n_retain']} samples):")
    print(f"  Ungated hit rate:  {eval_results['retain_ungated_rate']:.3f}")
    print(f"  Gated hit rate:    {eval_results['retain_gated_rate']:.3f}")

    eval_out = {k: v for k, v in eval_results.items() if k != "details"}
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"Eval results: {output_dir / 'eval_results.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
