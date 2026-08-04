#!/usr/bin/env python3
"""Export BeatNet model_1 weights to a stateless ONNX graph for Rust inference.

Usage:
  python scripts/export_beatnet_onnx.py [--weights PATH] [--out PATH] [--ref-audio PATH]

Downloads model_1_weights.pt from BeatNet if --weights is omitted.
Optionally dumps reference features/activations for a WAV (--ref-audio) when
BeatNet/madmom are installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "assets" / "beatnet" / "model_1.onnx"
WEIGHTS_URL = (
    "https://github.com/mjhydri/BeatNet/raw/main/src/BeatNet/models/model_1_weights.pt"
)


class BDA(nn.Module):
    """Beat/Downbeat Activation CRNN — mirrors BeatNet.model.BDA."""

    def __init__(self, dim_in: int, num_cells: int, num_layers: int, device: str = "cpu"):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hd = num_cells
        self.num_layers = num_layers
        self.conv_out = 150
        self.kernelsize = 10
        self.conv1 = nn.Conv1d(1, 2, self.kernelsize)
        self.linear0 = nn.Linear(
            2 * int((self.dim_in - self.kernelsize + 1) / 2), self.conv_out
        )
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=self.dim_hd,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = nn.Linear(in_features=self.dim_hd, out_features=3)
        self.to(device)

    def train_forward(self, data: torch.Tensor) -> torch.Tensor:
        """Stateless forward (no persistent LSTM hidden state)."""
        x = torch.reshape(data, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        x = self.linear0(x)
        x = torch.reshape(x, (data.shape[0], data.shape[1], self.conv_out))
        x = self.lstm(x)[0]
        out = self.linear(x)
        return out.transpose(1, 2)


class ExportWrapper(nn.Module):
    def __init__(self, model: BDA):
        super().__init__()
        self.model = model

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Softmax over class dim; matches final_pred on [3, T] (dim=0).
        out = self.model.train_forward(data)
        return torch.softmax(out, dim=1)


def ensure_weights(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights to {path} ...")
    urllib.request.urlretrieve(WEIGHTS_URL, path)
    return path


def export_onnx(weights: Path, out: Path, opset: int = 17) -> None:
    device = "cpu"
    model = BDA(272, 150, 2, device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 100, 272)
    out.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["features"],
        output_names=["activations"],
        dynamic_axes={"features": {1: "time"}, "activations": {2: "time"}},
        opset_version=opset,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(out), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(out), **export_kwargs)

    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def dump_reference(audio_path: Path, out_dir: Path) -> None:
    """Dump madmom LOG_SPECT features + BeatNet activations for golden tests."""
    try:
        from BeatNet.BeatNet import BeatNet
        from BeatNet.log_spect import LOG_SPECT
        import librosa
    except ImportError as e:
        print(f"Skipping reference dump (missing deps): {e}", file=sys.stderr)
        return

    sr = 22050
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    proc = LOG_SPECT(
        sample_rate=sr,
        win_length=int(64 * 0.001 * sr),
        hop_size=int(20 * 0.001 * sr),
        n_bands=[24],
        mode="offline",
    )
    feats = proc.process_audio(audio).T.astype(np.float32)  # [T, 272]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", feats)
    np.save(out_dir / "audio.npy", audio.astype(np.float32))

    bn = BeatNet(1, mode="offline", inference_model="DBN", device="cpu")
    # Raw CRNN activations (beat, downbeat) before DBN
    with torch.no_grad():
        t = torch.from_numpy(feats).unsqueeze(0)
        preds = bn.model.train_forward(t)[0]
        preds = bn.model.final_pred(preds).cpu().numpy()
        acts = np.transpose(preds[:2, :]).astype(np.float32)
    np.save(out_dir / "activations.npy", acts)
    print(f"Wrote reference dumps to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "assets" / "beatnet" / "model_1_weights.pt",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ref-audio", type=Path, default=None)
    p.add_argument(
        "--ref-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "beatnet" / "ref",
    )
    args = p.parse_args()

    weights = ensure_weights(args.weights)
    export_onnx(weights, args.out)
    if args.ref_audio is not None:
        dump_reference(args.ref_audio, args.ref_dir)


if __name__ == "__main__":
    main()
