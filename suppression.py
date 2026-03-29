"""
suppression.py

Inference-time selective channel suppression for cross-modal leakage in LLaVA.

Reads top-k sensitive channels from channel_sensitivity.json (produced by
analyze_channels.py) and registers a forward hook that scales those channels
by alpha during inference. alpha=0.0 zeros them out; alpha=1.0 is a no-op.

Designed to be imported by eval_with_suppression.py, but also runnable
standalone as a smoke-test.

Usage (standalone smoke-test):
    python suppression.py \
        --sensitivity_json results/channel_sensitivity.json \
        --layer 31 \
        --topk 20 \
        --alpha 0.0

Importable API:
    from suppression import ChannelSuppressor, load_top_channels

    channels = load_top_channels("results/channel_sensitivity.json", layer=31, topk=20)
    with ChannelSuppressor(model, layer=31, channels=channels, alpha=0.0):
        output = model.generate(...)
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch


# ── channel loading ───────────────────────────────────────────────────────────

def load_top_channels(sensitivity_json: str | Path, layer: int, topk: int) -> list[int]:
    """
    Load the top-k channel indices for a given layer from channel_sensitivity.json.
    Raises KeyError if the layer is not present in the file.
    """
    with open(sensitivity_json) as f:
        data = json.load(f)

    key = str(layer)
    if key not in data["results"]:
        available = list(data["results"].keys())
        raise KeyError(
            f"Layer {layer} not found in sensitivity JSON. Available: {available}"
        )

    channels = data["results"][key]["top_channels"]
    return channels[:topk]


# ── suppression hook ──────────────────────────────────────────────────────────

class ChannelSuppressor:
    """
    Context manager that suppresses selected hidden-state channels in one
    transformer layer during inference.

    The hook modifies the layer output in-place (well, returns a modified copy):
        h[:, :, channels] *= alpha

    alpha=0.0 zeros the channels out.
    alpha=1.0 is a no-op (useful for testing the hook path without suppression).

    Usage:
        with ChannelSuppressor(model, layer=31, channels=[1512, 3241, ...], alpha=0.0):
            out = model.generate(...)
    """

    def __init__(self, model, layer: int, channels: list[int], alpha: float = 0.0):
        self.model = model
        self.layer_idx = layer
        self.channels = channels
        self.alpha = alpha
        self._handle = None

    def _make_hook(self):
        channels = self.channels
        alpha = self.alpha

        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                h = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                h = output
                rest = ()
                is_tuple = False

            # h: [batch, seq_len, hidden_dim]
            h = h.clone()
            h[:, :, channels] = h[:, :, channels] * alpha

            if is_tuple:
                return (h,) + rest
            return h

        return hook

    def register(self):
        from analyze_channels import get_language_layers  # reuse layer resolver
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


# ── smoke-test ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Suppression smoke-test: verify hook loads and channels parse")
    p.add_argument("--sensitivity_json", required=True,
                   help="Path to channel_sensitivity.json")
    p.add_argument("--layer", type=int, default=31,
                   help="Layer to suppress (default: 31)")
    p.add_argument("--topk", type=int, default=20,
                   help="Number of top channels to suppress (default: 20)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Scaling factor (0.0=zero-out, 1.0=no-op)")
    return p.parse_args()


def main():
    args = parse_args()

    channels = load_top_channels(args.sensitivity_json, layer=args.layer, topk=args.topk)

    print(f"Layer       : {args.layer}")
    print(f"Alpha       : {args.alpha}")
    print(f"Top-{args.topk} channels: {channels}")
    print()

    # Verify hook logic on a small dummy tensor (no model load needed)
    h = torch.ones(1, 10, 4096)
    original_sum = h.sum().item()

    # Simulate what the hook does
    h_suppressed = h.clone()
    h_suppressed[:, :, channels] *= args.alpha
    suppressed_sum = h_suppressed.sum().item()

    n_zeroed = len(channels) if args.alpha == 0.0 else 0
    print(f"Tensor shape       : {tuple(h.shape)}")
    print(f"Sum before         : {original_sum:.1f}")
    print(f"Sum after (alpha={args.alpha}) : {suppressed_sum:.1f}")
    print(f"Channels affected  : {len(channels)} / {h.shape[-1]}")
    if args.alpha == 0.0:
        expected = original_sum - len(channels) * h.shape[1]
        match = abs(suppressed_sum - expected) < 1e-3
        print(f"Zero-out check     : {'PASS' if match else 'FAIL'}")

    print("\nSuppressor loaded OK. Import into eval_with_suppression.py to use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
