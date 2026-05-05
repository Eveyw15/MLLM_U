import sys
import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    import gating_module
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    gating_module = None


if torch is not None:
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))


    class TinyGate(nn.Module):
        def __init__(self, forget_gate=0.2, retain_gate=0.2):
            super().__init__()
            self.forget_logit = nn.Parameter(
                torch.logit(torch.tensor(float(forget_gate)))
            )
            self.retain_logit = nn.Parameter(
                torch.logit(torch.tensor(float(retain_gate)))
            )


class FakeGateHook:
    def __init__(self, *args, **kwargs):
        self.capture_gate_values = False
        self.last_gate_values = None
        self.last_gate_values_train = None
        self.registered = False
        self.removed = False

    def register(self):
        self.registered = True

    def remove(self):
        self.removed = True


@unittest.skipIf(torch is None, "torch is required for gating_module tests")
class GateModuleTests(unittest.TestCase):
    def test_parse_args_accepts_selectivity_options(self):
        argv = [
            "gating_module.py",
            "--smoke_test",
            "--lambda_selectivity",
            "0.2",
            "--selectivity_margin",
            "0.35",
        ]
        with patch.object(sys, "argv", argv):
            args = gating_module.parse_args()

        self.assertEqual(args.lambda_selectivity, 0.2)
        self.assertEqual(args.selectivity_margin, 0.35)

    def test_parse_args_accepts_gate_band_options(self):
        argv = [
            "gating_module.py",
            "--smoke_test",
            "--lambda_gate_band",
            "1.0",
            "--forget_gate_ceiling",
            "0.18",
            "--retain_gate_floor",
            "0.35",
        ]
        with patch.object(sys, "argv", argv):
            args = gating_module.parse_args()

        self.assertEqual(args.lambda_gate_band, 1.0)
        self.assertEqual(args.forget_gate_ceiling, 0.18)
        self.assertEqual(args.retain_gate_floor, 0.35)

    def test_parse_args_accepts_forget_weight(self):
        argv = [
            "gating_module.py",
            "--smoke_test",
            "--lambda_forget",
            "0.25",
        ]
        with patch.object(sys, "argv", argv):
            args = gating_module.parse_args()

        self.assertEqual(args.lambda_forget, 0.25)

    def run_paired_epoch_with_gate_values(
        self,
        forget_gate,
        retain_gate,
        forget_gate_ceiling=0.18,
        retain_gate_floor=0.35,
    ):
        gate = TinyGate(forget_gate=forget_gate, retain_gate=retain_gate)

        with patch.object(gating_module, "GateHook", FakeGateHook):
            trainer = gating_module.GateTrainer(
                TinyModel(),
                processor=None,
                gate=gate,
                layer=31,
                channel_indices=[1512],
                device="cpu",
                lr=0.0,
                lambda_retain=1.5,
                lambda_gate_retain=0.03,
                lambda_gate_forget=0.2,
                lambda_selectivity=0.2,
                selectivity_margin=0.2,
                lambda_gate_band=1.0,
                forget_gate_ceiling=forget_gate_ceiling,
                retain_gate_floor=retain_gate_floor,
            )

        def fake_forget_loss(model, processor, sample, device):
            values = torch.sigmoid(gate.forget_logit).reshape(1, 1)
            trainer.gate_hook.last_gate_values_train = values
            trainer.gate_hook.last_gate_values = values.detach().cpu()
            return values.sum() * 0.0 + 2.0

        def fake_retain_loss(model, processor, sample, device):
            values = torch.sigmoid(gate.retain_logit).reshape(1, 1)
            trainer.gate_hook.last_gate_values_train = values
            trainer.gate_hook.last_gate_values = values.detach().cpu()
            return values.sum() * 0.0 + 3.0

        with patch.object(gating_module, "compute_forget_loss", fake_forget_loss), \
             patch.object(gating_module, "compute_retain_loss", fake_retain_loss):
            return trainer.train_epoch_paired(
                forget_set=[{"id": "f1"}, {"id": "f2"}],
                retain_set=[{"id": "r1"}],
                epoch=0,
            )

    def test_paired_epoch_uses_train_gate_values_for_selectivity_stats(self):
        gate = TinyGate(forget_gate=0.12, retain_gate=0.12)

        with patch.object(gating_module, "GateHook", FakeGateHook):
            trainer = gating_module.GateTrainer(
                TinyModel(),
                processor=None,
                gate=gate,
                layer=31,
                channel_indices=[1512],
                device="cpu",
                lr=0.0,
                lambda_retain=1.5,
                lambda_gate_retain=0.03,
                lambda_gate_forget=0.2,
                lambda_selectivity=0.2,
                selectivity_margin=0.2,
                lambda_gate_band=0.0,
                forget_gate_ceiling=0.18,
                retain_gate_floor=0.0,
            )

        def fake_forget_loss(model, processor, sample, device):
            values = torch.sigmoid(gate.forget_logit).reshape(1, 1)
            trainer.gate_hook.last_gate_values_train = values
            trainer.gate_hook.last_gate_values = values.detach().cpu()
            return values.sum() * 0.0 + 2.0

        def fake_retain_loss(model, processor, sample, device):
            values = torch.sigmoid(gate.retain_logit).reshape(1, 1)
            trainer.gate_hook.last_gate_values_train = values
            trainer.gate_hook.last_gate_values = values.detach().cpu()
            return values.sum() * 0.0 + 3.0

        with patch.object(gating_module, "compute_forget_loss", fake_forget_loss), \
             patch.object(gating_module, "compute_retain_loss", fake_retain_loss):
            stats = trainer.train_epoch_paired(
                forget_set=[{"id": "f1"}, {"id": "f2"}],
                retain_set=[{"id": "r1"}],
                epoch=0,
            )

        gate_mean = 0.12
        self.assertEqual(stats["n_steps"], 2)
        self.assertEqual(stats["n_forget"], 2)
        self.assertEqual(stats["n_retain"], 2)
        self.assertAlmostEqual(stats["avg_forget_loss"], 2.0)
        self.assertAlmostEqual(stats["avg_retain_loss"], 3.0)
        self.assertAlmostEqual(stats["avg_forget_gate_mean"], gate_mean, places=5)
        self.assertAlmostEqual(stats["avg_retain_gate_mean"], gate_mean, places=5)
        self.assertAlmostEqual(stats["avg_gate_gap"], 0.0)
        self.assertAlmostEqual(stats["avg_selectivity_loss"], 0.2)
        self.assertIsNotNone(gate.forget_logit.grad)
        self.assertIsNotNone(gate.retain_logit.grad)

    def test_gate_band_loss_is_zero_inside_target_band(self):
        stats = self.run_paired_epoch_with_gate_values(
            forget_gate=0.12,
            retain_gate=0.42,
            forget_gate_ceiling=0.18,
            retain_gate_floor=0.35,
        )

        self.assertAlmostEqual(stats["avg_forget_ceiling_loss"], 0.0)
        self.assertAlmostEqual(stats["avg_retain_floor_loss"], 0.0)
        self.assertAlmostEqual(stats["avg_gate_band_loss"], 0.0)
        self.assertEqual(stats["forget_gate_ceiling"], 0.18)
        self.assertEqual(stats["retain_gate_floor"], 0.35)

    def test_gate_band_loss_is_positive_outside_target_band(self):
        stats = self.run_paired_epoch_with_gate_values(
            forget_gate=0.28,
            retain_gate=0.20,
            forget_gate_ceiling=0.18,
            retain_gate_floor=0.35,
        )

        self.assertGreater(stats["avg_forget_ceiling_loss"], 0.0)
        self.assertGreater(stats["avg_retain_floor_loss"], 0.0)
        self.assertGreater(stats["avg_gate_band_loss"], 0.0)

    def test_lambda_forget_scales_paired_forget_gradient(self):
        forget_gate = 0.4
        gate = TinyGate(forget_gate=forget_gate, retain_gate=0.4)

        with patch.object(gating_module, "GateHook", FakeGateHook):
            trainer = gating_module.GateTrainer(
                TinyModel(),
                processor=None,
                gate=gate,
                layer=31,
                channel_indices=[1512],
                device="cpu",
                lr=0.0,
                lambda_forget=0.25,
                lambda_retain=0.0,
                lambda_gate_retain=0.0,
                lambda_gate_forget=0.0,
                lambda_selectivity=0.0,
                selectivity_margin=0.0,
                lambda_gate_band=0.0,
            )

        def fake_forget_loss(model, processor, sample, device):
            values = torch.sigmoid(gate.forget_logit).reshape(1, 1)
            trainer.gate_hook.last_gate_values_train = values
            return values.mean()

        def fake_retain_loss(model, processor, sample, device):
            values = torch.sigmoid(gate.retain_logit).reshape(1, 1)
            trainer.gate_hook.last_gate_values_train = values
            return values.mean() * 0.0

        with patch.object(gating_module, "compute_forget_loss", fake_forget_loss), \
             patch.object(gating_module, "compute_retain_loss", fake_retain_loss):
            stats = trainer.train_epoch_paired(
                forget_set=[{"id": "f1"}],
                retain_set=[{"id": "r1"}],
                epoch=0,
            )

        expected_grad = 0.25 * forget_gate * (1.0 - forget_gate)
        self.assertAlmostEqual(gate.forget_logit.grad.item(), expected_grad)
        self.assertEqual(stats["lambda_forget"], 0.25)


if __name__ == "__main__":
    unittest.main()
