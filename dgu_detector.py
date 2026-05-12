"""
dgu_detector.py

Detector-Guided Suppression (DGU-lite) for cross-modal leakage experiments.

This script trains a small binary detector on rich pooled activations from the
top-k sensitive channels at a target LLaVA language layer. At inference time,
the detector leak score controls a scalar suppression gate for those channels:

    leak_score = sigmoid(detector(pool_rich_stats(h_k)))
    scalar_gate = alpha_retain + leak_score * (alpha_forget - alpha_retain)
    h[:, :, top_k_channels] *= scalar_gate

The LLaVA model is frozen. Only detector parameters are trained with paired
BCEWithLogitsLoss plus an optional leak-score separation loss.
"""

import argparse
import json
import math
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from gating_module import split_forget_retain

try:
    from gating_module import pool_rich_stats
except ImportError:
    def pool_rich_stats(h_k: torch.Tensor) -> torch.Tensor:
        """Fallback rich pooling if running against an older gating_module.py."""
        h_float = h_k.float()
        abs_h = h_float.abs()
        abs_mean = abs_h.mean(dim=1)
        signed_mean = h_float.mean(dim=1)
        std = h_float.std(dim=1, unbiased=False)
        max_abs = abs_h.amax(dim=1)
        return torch.cat([abs_mean, signed_mean, std, max_abs], dim=-1)


def compute_scalar_gate(leak_score: torch.Tensor,
                        alpha_forget: float,
                        alpha_retain: float) -> torch.Tensor:
    """
    Convert detector leak scores into scalar channel gates.

    leak_score=1 maps to alpha_forget.
    leak_score=0 maps to alpha_retain.
    """
    hard_leak = (leak_score >= score_threshold).to(leak_score.dtype)
    return alpha_retain + hard_leak * (alpha_forget - alpha_retain)


def build_prompt_only_inputs(processor, image, question: str, device: str):
    """
    Build prompt-only LLaVA inputs for detector training.

    DGU decides whether to suppress before answer generation, so the detector
    is trained on the same prompt-only state it will see at inference time.
    """
    from minimal_eval_unlok import build_prompt

    prompt = build_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return inputs.to(device)


class LeakDetector(nn.Module):
    """Small MLP leak detector over normalized rich top-k channel statistics."""

    def __init__(self, k: int, hidden_dim: int = 16):
        super().__init__()
        self.k = k
        self.input_dim = 4 * k
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, 4k]
        Returns:
            logits: [B, 1]
        """
        x = self.norm(features.to(torch.float32))
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward_hk(self, h_k: torch.Tensor) -> torch.Tensor:
        """Pool [B, T, k] activations and return detector logits [B, 1]."""
        return self.forward(pool_rich_stats(h_k))


class DGUHook:
    """
    Forward hook that applies detector-guided scalar suppression.

    It stores the detector logits for training and leak_score/scalar_gate values
    for diagnostics. The selected channel activations are detached before
    feature pooling so gradients are confined to detector parameters.
    """

    def __init__(self, model, detector: LeakDetector,
             layer: int, channel_indices: list[int],
             alpha_forget: float = 0.0,
             alpha_retain: float = 1.0,
             capture_values: bool = False,
             freeze_gate_after_first: bool = True,
             suppression_mode: str = "continuous",
             score_threshold: float = 0.5):
        self.model = model
        self.detector = detector
        self.layer_idx = layer
        self.channel_indices = channel_indices
        self.alpha_forget = alpha_forget
        self.alpha_retain = alpha_retain
        self.capture_values = capture_values
        self.freeze_gate_after_first = freeze_gate_after_first
        self.suppression_mode = suppression_mode
        self.score_threshold = score_threshold
        self.apply_suppression = apply_suppression

        self._handle = None
        self.last_detector_logits_train = None
        self.last_leak_score_train = None
        self.last_scalar_gate_train = None
        self.last_leak_score = None
        self.last_scalar_gate = None
        self.leak_score_history = []
        self.scalar_gate_history = []
        self.frozen_scalar_gate = None
        self.frozen_leak_score = None
        self.n_detector_forwards = 0

    def reset_train_values(self):
        self.last_detector_logits_train = None
        self.last_leak_score_train = None
        self.last_scalar_gate_train = None

    def reset_frozen_gate(self):
        self.frozen_scalar_gate = None
        self.frozen_leak_score = None

    def _make_hook(self):
        detector = self.detector
        ch_idx = self.channel_indices
        hook_self = self

        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                h = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                h = output
                rest = ()
                is_tuple = False

            h_k = h[:, :, ch_idx]
            if (
                hook_self.freeze_gate_after_first
                and hook_self.frozen_scalar_gate is not None
            ):
                leak_score = hook_self.frozen_leak_score
                scalar_gate = hook_self.frozen_scalar_gate
            else:
                features = pool_rich_stats(h_k.detach())
                logits = detector(features)
                hook_self.n_detector_forwards += 1
                leak_score = torch.sigmoid(logits)

                if hook_self.suppression_mode == "continuous":
                    scalar_gate = compute_scalar_gate(
                        leak_score,
                        hook_self.alpha_forget,
                        hook_self.alpha_retain,
                    )
                elif hook_self.suppression_mode == "threshold":
                    scalar_gate = compute_thresholded_scalar_gate(
                        leak_score,
                        hook_self.alpha_forget,
                        hook_self.alpha_retain,
                        hook_self.score_threshold,
                    )
                else:
                    raise ValueError(f"Unknown suppression_mode: {hook_self.suppression_mode}")

                hook_self.last_detector_logits_train = logits
                if hook_self.freeze_gate_after_first:
                    hook_self.frozen_leak_score = leak_score.detach()
                    hook_self.frozen_scalar_gate = scalar_gate.detach()

            hook_self.last_leak_score_train = leak_score
            hook_self.last_scalar_gate_train = scalar_gate

            if hook_self.capture_values:
                leak_cpu = leak_score.detach().cpu().float()
                gate_cpu = scalar_gate.detach().cpu().float()
                hook_self.last_leak_score = leak_cpu
                hook_self.last_scalar_gate = gate_cpu
                hook_self.leak_score_history.append(leak_cpu)
                hook_self.scalar_gate_history.append(gate_cpu)

            if hook_self.apply_suppression:
                h = h.clone()
                gate_for_apply = scalar_gate.detach().to(
                    device=h_k.device, dtype=h_k.dtype
                ).unsqueeze(1)
                h[:, :, ch_idx] = h_k * gate_for_apply

            if is_tuple:
                return (h,) + rest
            return h

        return hook_fn

    def register(self):
        self.leak_score_history = []
        self.scalar_gate_history = []
        self.last_leak_score = None
        self.last_scalar_gate = None
        self.n_detector_forwards = 0
        self.reset_train_values()
        self.reset_frozen_gate()

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


def summarize_dgu_history(leak_score_history: list[torch.Tensor],
                          scalar_gate_history: list[torch.Tensor]) -> dict:
    """Summarize detector/gate values captured across hook calls."""
    if not leak_score_history or not scalar_gate_history:
        nan = float("nan")
        return {
            "leak_score": nan,
            "scalar_gate": nan,
            "mean_leak_score": nan,
            "mean_scalar_gate": nan,
            "first_leak_score": nan,
            "last_leak_score": nan,
            "first_scalar_gate": nan,
            "last_scalar_gate": nan,
            "n_hook_calls": 0,
            "n_detector_forwards": 0,
        }

    leak_stack = torch.stack(leak_score_history, dim=0)
    gate_stack = torch.stack(scalar_gate_history, dim=0)
    first_leak_score = leak_stack[0].mean().item()
    last_leak_score = leak_stack[-1].mean().item()
    mean_leak_score = leak_stack.mean().item()
    first_scalar_gate = gate_stack[0].mean().item()
    last_scalar_gate = gate_stack[-1].mean().item()
    mean_scalar_gate = gate_stack.mean().item()
    n_hook_calls = len(leak_score_history)

    return {
        "leak_score": first_leak_score,
        "scalar_gate": first_scalar_gate,
        "mean_leak_score": mean_leak_score,
        "mean_scalar_gate": mean_scalar_gate,
        "first_leak_score": first_leak_score,
        "last_leak_score": last_leak_score,
        "first_scalar_gate": first_scalar_gate,
        "last_scalar_gate": last_scalar_gate,
        "n_hook_calls": n_hook_calls,
        "n_detector_forwards": n_hook_calls,
    }


class DGUTrainer:
    """Train, evaluate, and save a detector-guided suppression model."""

    def __init__(self, model, processor,
             detector: LeakDetector,
             layer: int,
             channel_indices: list[int],
             device: str,
             lr: float = 0.001,
             lambda_sep: float = 1.0,
             score_margin: float = 0.3,
             alpha_forget: float = 0.0,
             alpha_retain: float = 1.0,
             freeze_gate_after_first: bool = True,
             suppression_mode: str = "continuous",
             score_threshold: float = 0.5):
        self.model = model
        self.processor = processor
        self.detector = detector
        self.layer = layer
        self.channel_indices = channel_indices
        self.device = device
        self.lambda_sep = lambda_sep
        self.score_margin = score_margin
        self.alpha_forget = alpha_forget
        self.alpha_retain = alpha_retain
        self.freeze_gate_after_first = freeze_gate_after_first
        self.suppression_mode = suppression_mode
        self.score_threshold = score_threshold

        for p in model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        # Training only needs detector logits. It should not alter the frozen
        # LLaVA hidden states, because the training loss is computed directly
        # from the captured detector logits rather than from model outputs.
        self.hook = DGUHook(
            model, detector, layer, channel_indices,
            alpha_forget=alpha_forget,
            alpha_retain=alpha_retain,
            freeze_gate_after_first=False,
            suppression_mode="continuous",
            score_threshold=score_threshold,
            apply_suppression=False,
        )

    def build_training_inputs(self, sample: dict):
        """Build prompt-only inputs for detector training."""
        from PIL import Image

        image = Image.open(sample["image_path"]).convert("RGB")
        return build_prompt_only_inputs(
            self.processor, image, sample["question"], self.device
        )

    def run_detector_forward(self, sample: dict) -> torch.Tensor:
        """Run one sample through the frozen model and return detector logits."""
        self.hook.reset_train_values()
        inputs = self.build_training_inputs(sample)
        outputs = self.model(**inputs)
        del outputs

        logits = self.hook.last_detector_logits_train
        if logits is None:
            raise RuntimeError("Detector hook did not capture logits")
        return logits.float()

    def train_epoch(self, forget_set: list[dict],
                    retain_set: list[dict],
                    epoch: int) -> dict:
        """Run one balanced paired detector-training epoch."""
        self.model.eval()
        self.detector.train()
        self.hook.capture_values = False
        self.hook.register()

        n_steps = max(len(forget_set), len(retain_set))
        total_bce_loss = 0.0
        total_sep_loss = 0.0
        total_loss = 0.0
        total_forget_score = 0.0
        total_retain_score = 0.0
        total_score_gap = 0.0
        total_forget_gate = 0.0
        total_retain_gate = 0.0
        n_success = 0
        n_fail = 0
        printed_traceback = False

        try:
            pbar = tqdm(range(n_steps), desc=f"Epoch {epoch + 1}")
            for _ in pbar:
                forget_sample = random.choice(forget_set)
                retain_sample = random.choice(retain_set)
                self.optimizer.zero_grad(set_to_none=True)

                try:
                    forget_logits = self.run_detector_forward(forget_sample)
                    retain_logits = self.run_detector_forward(retain_sample)

                    forget_labels = torch.ones_like(forget_logits)
                    retain_labels = torch.zeros_like(retain_logits)
                    forget_bce = self.criterion(forget_logits, forget_labels)
                    retain_bce = self.criterion(retain_logits, retain_labels)
                    bce_loss = 0.5 * (forget_bce + retain_bce)

                    forget_score = torch.sigmoid(forget_logits)
                    retain_score = torch.sigmoid(retain_logits)
                    score_gap = forget_score.mean() - retain_score.mean()
                    sep_loss = F.relu(
                        forget_logits.new_tensor(self.score_margin) - score_gap
                    )

                    loss = bce_loss + self.lambda_sep * sep_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.detector.parameters(), 1.0)
                    self.optimizer.step()

                    with torch.no_grad():
                        forget_score_val = forget_score.mean().item()
                        retain_score_val = retain_score.mean().item()
                        score_gap_val = forget_score_val - retain_score_val
                        forget_gate = compute_scalar_gate(
                            forget_score,
                            self.alpha_forget,
                            self.alpha_retain,
                        ).mean().item()
                        retain_gate = compute_scalar_gate(
                            retain_score,
                            self.alpha_forget,
                            self.alpha_retain,
                        ).mean().item()

                    total_bce_loss += bce_loss.item()
                    total_sep_loss += sep_loss.item()
                    total_loss += loss.item()
                    total_forget_score += forget_score_val
                    total_retain_score += retain_score_val
                    total_score_gap += score_gap_val
                    total_forget_gate += forget_gate
                    total_retain_gate += retain_gate
                    n_success += 1

                except Exception as e:
                    n_fail += 1
                    if not printed_traceback:
                        traceback.print_exc()
                        printed_traceback = True
                    print(f"  Error on pair: {e}")
                    continue

                pbar.set_postfix({
                    "bce": f"{total_bce_loss / max(n_success, 1):.3f}",
                    "sep": f"{total_sep_loss / max(n_success, 1):.3f}",
                    "gap": f"{total_score_gap / max(n_success, 1):+.3f}",
                })
        finally:
            self.hook.remove()

        avg_forget_score = total_forget_score / max(n_success, 1)
        avg_retain_score = total_retain_score / max(n_success, 1)
        avg_score_gap = total_score_gap / max(n_success, 1)
        avg_forget_gate = total_forget_gate / max(n_success, 1)
        avg_retain_gate = total_retain_gate / max(n_success, 1)

        return {
            "epoch": epoch + 1,
            "avg_loss": total_loss / max(n_success, 1),
            "avg_bce_loss": total_bce_loss / max(n_success, 1),
            "avg_sep_loss": total_sep_loss / max(n_success, 1),
            "avg_forget_score": avg_forget_score,
            "avg_retain_score": avg_retain_score,
            "avg_score_gap": avg_score_gap,
            "avg_forget_leak_score": avg_forget_score,
            "avg_retain_leak_score": avg_retain_score,
            "forget_minus_retain_leak_score_gap": avg_score_gap,
            "retain_minus_forget_leak_score_gap": -avg_score_gap,
            "avg_forget_scalar_gate": avg_forget_gate,
            "avg_retain_scalar_gate": avg_retain_gate,
            "lambda_sep": self.lambda_sep,
            "score_margin": self.score_margin,
            "n_steps": n_steps,
            "n_success": n_success,
            "n_forget": n_success,
            "n_retain": n_success,
            "n_fail": n_fail,
        }

    def evaluate(self, forget_set: list[dict],
                 retain_set: list[dict]) -> dict:
        """Measure DGU-gated vs ungated hit rates and detector diagnostics."""
        from PIL import Image
        from minimal_eval_unlok import clean_answer, run_with_image, soft_match

        self.model.eval()
        self.detector.eval()
        results = {"forget": [], "retain": []}

        for split_name, samples in [("forget", forget_set), ("retain", retain_set)]:
            for sample in tqdm(samples, desc=f"Eval {split_name}"):
                img = Image.open(sample["image_path"]).convert("RGB")
                target = sample["target"]

                raw_ungated = run_with_image(
                    self.model, self.processor,
                    sample["question"], img, self.device,
                )
                ungated_answer = clean_answer(raw_ungated)
                ungated_match = soft_match(target, ungated_answer)

                diag_hook = DGUHook(
                    self.model, self.detector, self.layer,
                    self.channel_indices,
                    alpha_forget=self.alpha_forget,
                    alpha_retain=self.alpha_retain,
                    capture_values=True,
                    freeze_gate_after_first=self.freeze_gate_after_first,
                    suppression_mode=self.suppression_mode,
                    score_threshold=self.score_threshold,
                    apply_suppression=True,
                )
                with diag_hook:
                    raw_gated = run_with_image(
                        self.model, self.processor,
                        sample["question"], img, self.device,
                    )
                gated_answer = clean_answer(raw_gated)
                gated_match = soft_match(target, gated_answer)

                if ungated_match and not gated_match:
                    flip_type = "good_flip"
                elif not ungated_match and gated_match:
                    flip_type = "bad_flip"
                else:
                    flip_type = "unchanged"

                leak_history = diag_hook.leak_score_history
                gate_history = diag_hook.scalar_gate_history
                if not leak_history and diag_hook.last_leak_score is not None:
                    leak_history = [diag_hook.last_leak_score]
                if not gate_history and diag_hook.last_scalar_gate is not None:
                    gate_history = [diag_hook.last_scalar_gate]
                dgu_summary = summarize_dgu_history(leak_history, gate_history)
                dgu_summary["n_detector_forwards"] = diag_hook.n_detector_forwards

                results[split_name].append({
                    "id": sample["id"],
                    "target": target,
                    "ungated_match": ungated_match,
                    "gated_match": gated_match,
                    "flip_type": flip_type,
                    "leak_score": dgu_summary["leak_score"],
                    "scalar_gate": dgu_summary["scalar_gate"],
                    "mean_leak_score": dgu_summary["mean_leak_score"],
                    "mean_scalar_gate": dgu_summary["mean_scalar_gate"],
                    "ungated_answer": ungated_answer,
                    "gated_answer": gated_answer,
                    **dgu_summary,
                })

        def rate(records, key):
            vals = [r[key] for r in records]
            return sum(vals) / len(vals) if vals else float("nan")

        def mean_value(records, key):
            vals = [r[key] for r in records if not math.isnan(r[key])]
            return sum(vals) / len(vals) if vals else float("nan")

        forget_records = results["forget"]
        retain_records = results["retain"]
        forget_leak_score = mean_value(forget_records, "leak_score")
        retain_leak_score = mean_value(retain_records, "leak_score")

        return {
            "forget_ungated_rate": round(rate(forget_records, "ungated_match"), 4),
            "forget_gated_rate": round(rate(forget_records, "gated_match"), 4),
            "forget_good_flips": sum(
                1 for r in forget_records if r["flip_type"] == "good_flip"
            ),
            "forget_bad_flips": sum(
                1 for r in forget_records if r["flip_type"] == "bad_flip"
            ),
            "forget_mean_leak_score": round(forget_leak_score, 4),
            "forget_mean_scalar_gate": round(mean_value(forget_records, "scalar_gate"), 4),
            "retain_ungated_rate": round(rate(retain_records, "ungated_match"), 4),
            "retain_gated_rate": round(rate(retain_records, "gated_match"), 4),
            "retain_mean_leak_score": round(retain_leak_score, 4),
            "retain_mean_scalar_gate": round(mean_value(retain_records, "scalar_gate"), 4),
            "retain_minus_forget_leak_score_gap": round(
                retain_leak_score - forget_leak_score, 4
            ),
            "forget_minus_retain_leak_score_gap": round(
                forget_leak_score - retain_leak_score, 4
            ),
            "n_forget": len(forget_records),
            "n_retain": len(retain_records),
            "details": results,
        }

    def build_diagnostics(self, eval_results: dict) -> dict:
        """Build detailed detector diagnostics for JSON output."""
        diag = {
            "channel_indices": self.channel_indices,
            "layer": self.layer,
            "alpha_forget": self.alpha_forget,
            "alpha_retain": self.alpha_retain,
            "freeze_gate_after_first": self.freeze_gate_after_first,
            "suppression_mode": self.suppression_mode,
            "score_threshold": self.score_threshold,
            "splits": {},
        }

        for split_name in ["forget", "retain"]:
            records = eval_results["details"][split_name]
            leak_scores = np.array([r["leak_score"] for r in records], dtype=np.float64)
            scalar_gates = np.array([r["scalar_gate"] for r in records], dtype=np.float64)

            if len(records) == 0:
                continue

            diag["splits"][split_name] = {
                "n_samples": len(records),
                "mean_leak_score": float(np.nanmean(leak_scores)),
                "std_leak_score": float(np.nanstd(leak_scores)),
                "mean_scalar_gate": float(np.nanmean(scalar_gates)),
                "std_scalar_gate": float(np.nanstd(scalar_gates)),
                "samples": [
                    {
                        "id": r["id"],
                        "leak_score": round(r["leak_score"], 4),
                        "scalar_gate": round(r["scalar_gate"], 4),
                        "mean_leak_score": round(r["mean_leak_score"], 4),
                        "mean_scalar_gate": round(r["mean_scalar_gate"], 4),
                        "first_leak_score": round(r["first_leak_score"], 4),
                        "last_leak_score": round(r["last_leak_score"], 4),
                        "first_scalar_gate": round(r["first_scalar_gate"], 4),
                        "last_scalar_gate": round(r["last_scalar_gate"], 4),
                        "n_hook_calls": r["n_hook_calls"],
                        "n_detector_forwards": r["n_detector_forwards"],
                        "flip_type": r["flip_type"],
                        "ungated_answer": r["ungated_answer"],
                        "gated_answer": r["gated_answer"],
                    }
                    for r in records
                ],
            }

        f_info = diag["splits"].get("forget")
        r_info = diag["splits"].get("retain")
        if f_info and r_info:
            diag["retain_minus_forget_leak_score_gap"] = (
                r_info["mean_leak_score"] - f_info["mean_leak_score"]
            )
            diag["forget_minus_retain_leak_score_gap"] = (
                f_info["mean_leak_score"] - r_info["mean_leak_score"]
            )

        return diag

    def save(self, output_dir: Path):
        """Save detector weights and DGU config."""
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.detector.state_dict(),
            "layer": self.layer,
            "channel_indices": self.channel_indices,
            "k": self.detector.k,
            "hidden_dim": self.detector.hidden_dim,
            "input_dim": self.detector.input_dim,
            "pooling_mode": "rich",
            "alpha_forget": self.alpha_forget,
            "alpha_retain": self.alpha_retain,
            "lambda_sep": self.lambda_sep,
            "score_margin": self.score_margin,
            "freeze_gate_after_first": self.freeze_gate_after_first,
            "training_hook_applies_suppression": False,
            "eval_hook_applies_suppression": True,
            "suppression_mode": self.suppression_mode,
            "score_threshold": self.score_threshold,
            "training_mode": "prompt_only",
        }, output_dir / "detector_weights.pt")
        print(f"Detector saved to: {output_dir / 'detector_weights.pt'}")


def print_eval_summary(eval_results: dict):
    """Print the requested DGU evaluation summary."""
    print(f"\nForget set ({eval_results['n_forget']} samples):")
    print(f"  Ungated hit rate:       {eval_results['forget_ungated_rate']:.3f}")
    print(f"  DGU-gated hit rate:     {eval_results['forget_gated_rate']:.3f}")
    print(f"  Good flips (T->F):      {eval_results['forget_good_flips']}")
    print(f"  Bad flips (F->T):       {eval_results['forget_bad_flips']}")
    print(f"  Mean leak_score:        {eval_results['forget_mean_leak_score']:.4f}")
    print(f"  Mean scalar_gate:       {eval_results['forget_mean_scalar_gate']:.4f}")

    print(f"\nRetain set ({eval_results['n_retain']} samples):")
    print(f"  Ungated hit rate:       {eval_results['retain_ungated_rate']:.3f}")
    print(f"  DGU-gated hit rate:     {eval_results['retain_gated_rate']:.3f}")
    print(f"  Mean leak_score:        {eval_results['retain_mean_leak_score']:.4f}")
    print(f"  Mean scalar_gate:       {eval_results['retain_mean_scalar_gate']:.4f}")

    gap = eval_results["forget_minus_retain_leak_score_gap"]
    print(f"\nscore gap (forget - retain leak_score): {gap:+.4f}")
    print("desired: positive gap, because forget leak_score should exceed retain")


def smoke_test():
    """Run lightweight synthetic checks for detector and alpha rule."""
    print("=" * 60)
    print("  SMOKE TEST: DGU Detector")
    print("=" * 60)

    B, T, k = 3, 7, 5
    hidden_dim = 16
    h_k = torch.randn(B, T, k)
    detector = LeakDetector(k=k, hidden_dim=hidden_dim)

    features = pool_rich_stats(h_k)
    assert features.shape == (B, 4 * k), (
        f"Expected features {(B, 4 * k)}, got {features.shape}"
    )
    print(f"[PASS] Rich features shape: {features.shape}")

    logits = detector(features)
    assert logits.shape == (B, 1), (
        f"Expected detector output {(B, 1)}, got {logits.shape}"
    )
    print(f"[PASS] Detector output shape: {logits.shape}")

    leak_score = torch.sigmoid(logits)
    scalar_gate = compute_scalar_gate(
        leak_score, alpha_forget=0.0, alpha_retain=1.0
    )
    assert scalar_gate.shape == (B, 1), (
        f"Expected scalar gate {(B, 1)}, got {scalar_gate.shape}"
    )
    applied = h_k * scalar_gate.unsqueeze(1)
    assert applied.shape == h_k.shape, (
        f"Gate is not broadcastable: got {applied.shape}, expected {h_k.shape}"
    )
    print(f"[PASS] Scalar gate broadcast shape: {scalar_gate.unsqueeze(1).shape}")

    alpha_forget = 0.2
    alpha_retain = 0.9
    ones = torch.ones(B, 1)
    zeros = torch.zeros(B, 1)
    gate_forget = compute_scalar_gate(ones, alpha_forget, alpha_retain)
    gate_retain = compute_scalar_gate(zeros, alpha_forget, alpha_retain)
    assert torch.allclose(gate_forget, torch.full((B, 1), alpha_forget))
    assert torch.allclose(gate_retain, torch.full((B, 1), alpha_retain))
    print("[PASS] Alpha rule: leak_score=1 -> alpha_forget")
    print("[PASS] Alpha rule: leak_score=0 -> alpha_retain")

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a DGU-lite leak detector for selective suppression"
    )
    p.add_argument("--data_path",
                   help="Path to zsre_mend_eval.json")
    p.add_argument("--coco_root",
                   help="Root of COCO images (train2017/ and val2017/)")
    p.add_argument("--baseline_jsonl",
                   help="Output of minimal_eval_unlok.py")
    p.add_argument("--sensitivity_json",
                   help="channel_sensitivity.json from analyze_channels.py")
    p.add_argument("--output_dir",
                   help="Directory to save detector weights and logs")
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--lambda_sep", type=float, default=1.0,
                   help="Weight for paired leak-score separation loss")
    p.add_argument("--score_margin", type=float, default=0.3,
                   help="Desired forget minus retain leak-score margin")
    p.add_argument("--alpha_forget", type=float, default=0.0)
    p.add_argument("--alpha_retain", type=float, default=1.0)
    p.add_argument("--suppression_mode",
                choices=["continuous", "threshold"],
                default="continuous",
                help="How leak_score is converted into scalar gate")
    p.add_argument("--score_threshold", type=float, default=0.5,
                help="Threshold for hard DGU suppression when suppression_mode=threshold")
    p.add_argument("--no_freeze_gate_after_first", action="store_true",
                   help="Recompute DGU gate on every generation forward")
    p.add_argument("--no_quantize", action="store_true",
                   help="Disable 8-bit quantization")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run smoke test only")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        smoke_test()
        return 0

    missing = [
        name for name, val in [
            ("--data_path", args.data_path),
            ("--coco_root", args.coco_root),
            ("--baseline_jsonl", args.baseline_jsonl),
            ("--sensitivity_json", args.sensitivity_json),
            ("--output_dir", args.output_dir),
        ] if val is None
    ]
    if missing:
        import sys
        print(f"error: the following arguments are required for training: {', '.join(missing)}")
        sys.exit(2)

    from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
    from analyze_channels import get_language_layers
    from minimal_eval_unlok import load_processor
    from suppression import load_top_channels

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

    forget_set, retain_set = split_forget_retain(baseline_jsonl, data_path, coco_root)
    print(f"Forget set (leak-like): {len(forget_set)}")
    print(f"Retain set (non-leak):  {len(retain_set)}")

    if len(forget_set) == 0 or len(retain_set) == 0:
        print("Both forget and retain samples are required for detector training.")
        return 1

    channel_indices = load_top_channels(
        sensitivity_json, layer=args.layer, topk=args.topk
    )
    print(f"Layer {args.layer}, top-{args.topk} channels: {channel_indices}")

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

    k = len(channel_indices)
    detector = LeakDetector(k=k, hidden_dim=args.hidden_dim)

    layers = get_language_layers(model)
    layer_device = next(layers[args.layer].parameters()).device
    detector = detector.to(device=layer_device, dtype=torch.float32)

    n_params = sum(p.numel() for p in detector.parameters())
    print(f"Detector created: input_dim={detector.input_dim}, "
          f"hidden_dim={detector.hidden_dim}, {n_params} trainable parameters")
    print(f"Alpha rule: forget={args.alpha_forget}, retain={args.alpha_retain}")
    freeze_gate_after_first = not args.no_freeze_gate_after_first
    print(f"Freeze gate after first generation forward: {freeze_gate_after_first}")
    print(f"Suppression mode: {args.suppression_mode}")
    if args.suppression_mode == "threshold":
        print(f"Score threshold: {args.score_threshold}")

    trainer = DGUTrainer(
        model, processor, detector,
        layer=args.layer,
        channel_indices=channel_indices,
        device=device,
        lr=args.lr,
        lambda_sep=args.lambda_sep,
        score_margin=args.score_margin,
        alpha_forget=args.alpha_forget,
        alpha_retain=args.alpha_retain,
        freeze_gate_after_first=freeze_gate_after_first,
        suppression_mode=args.suppression_mode,
        score_threshold=args.score_threshold,
    )

    log = []
    for epoch in range(args.epochs):
        stats = trainer.train_epoch(forget_set, retain_set, epoch)
        log.append(stats)
        print(f"\n  Epoch {stats['epoch']}: "
              f"bce={stats['avg_bce_loss']:.4f}  "
              f"sep={stats['avg_sep_loss']:.4f}  "
              f"forget_score={stats['avg_forget_score']:.4f}  "
              f"retain_score={stats['avg_retain_score']:.4f}  "
              f"score_gap={stats['avg_score_gap']:+.4f}  "
              f"failures={stats['n_fail']}")

    trainer.save(output_dir)

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log: {output_dir / 'training_log.json'}")

    print("\n===== POST-TRAINING DGU EVALUATION =====")
    eval_results = trainer.evaluate(forget_set, retain_set)
    print_eval_summary(eval_results)

    eval_out = {k: v for k, v in eval_results.items() if k != "details"}
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"Eval results: {output_dir / 'eval_results.json'}")

    diagnostics = trainer.build_diagnostics(eval_results)
    diagnostics["config"] = {
        "model_id": args.model_id,
        "layer": args.layer,
        "topk": args.topk,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "lambda_sep": args.lambda_sep,
        "score_margin": args.score_margin,
        "alpha_forget": args.alpha_forget,
        "alpha_retain": args.alpha_retain,
        "freeze_gate_after_first": freeze_gate_after_first,
        "suppression_mode": args.suppression_mode,
        "score_threshold": args.score_threshold,
        "training_hook_applies_suppression": False,
        "eval_hook_applies_suppression": True,
        "pooling_mode": "rich",
        "training_mode": "prompt_only",
    }
    with open(output_dir / "dgu_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"DGU diagnostics: {output_dir / 'dgu_diagnostics.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
