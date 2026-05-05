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
        def __init__(self):
            super().__init__()
            self.forget_logit = nn.Parameter(torch.tensor(-2.0))
            self.retain_logit = nn.Parameter(torch.tensor(-2.0))


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

    def test_paired_epoch_uses_train_gate_values_for_selectivity_stats(self):
        gate = TinyGate()

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

        gate_mean = torch.sigmoid(torch.tensor(-2.0)).item()
        self.assertEqual(stats["n_steps"], 2)
        self.assertEqual(stats["n_forget"], 2)
        self.assertEqual(stats["n_retain"], 2)
        self.assertAlmostEqual(stats["avg_forget_loss"], 2.0)
        self.assertAlmostEqual(stats["avg_retain_loss"], 3.0)
        self.assertAlmostEqual(stats["avg_forget_gate_mean"], gate_mean)
        self.assertAlmostEqual(stats["avg_retain_gate_mean"], gate_mean)
        self.assertAlmostEqual(stats["avg_gate_gap"], 0.0)
        self.assertAlmostEqual(stats["avg_selectivity_loss"], 0.2)
        self.assertIsNotNone(gate.forget_logit.grad)
        self.assertIsNotNone(gate.retain_logit.grad)


if __name__ == "__main__":
    unittest.main()
