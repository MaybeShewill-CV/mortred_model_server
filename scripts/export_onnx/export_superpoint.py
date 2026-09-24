#!/usr/bin/env python3
"""P2.E SuperPoint 120x160 GRAY from MagicLeap pretrained weights."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPSET, write_pair_from_static  # noqa: E402

# Architecture from magicleap/SuperPointPretrainedNetwork (BSD).
class SuperPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = nn.Conv2d(1, c1, 3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, 3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, 3, stride=1, padding=1)
        self.convPa = nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, 1, stride=1, padding=0)
        self.convDa = nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)
        return semi, desc


def main() -> int:
    ckpt = Path("/tmp/mortred_p2/superpoint_v1.pth")
    net = SuperPointNet()
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    dummy = torch.randn(1, 1, 120, 160)
    tmp = Path("/tmp/mortred_p2/superpoint_120x160.export.onnx")
    torch.onnx.export(
        net,
        dummy,
        str(tmp),
        input_names=["input"],
        output_names=["output_1", "output_2"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    out_dir = ROOT / "weights" / "feature_point" / "superpoint"
    write_pair_from_static(
        tmp,
        out_dir / "superpoint_120x160.static_bs1.onnx",
        out_dir / "superpoint_120x160.dyn.onnx",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
