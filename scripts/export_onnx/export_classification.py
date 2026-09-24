#!/usr/bin/env python3
"""P2.B: torchvision classifiers wrapped as NHWC F32 (preprocess stays in C++)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPSET, write_pair_from_static  # noqa: E402


class NhwcWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C++ already did caffe mean/std RGB; graph only switches NHWC -> NCHW.
        return self.net(x.permute(0, 3, 1, 2).contiguous())


def export_one(ctor, weights_enum, stem: str) -> None:
    net = ctor(weights=weights_enum)
    net.eval()
    wrapped = NhwcWrapper(net).eval()
    dummy = torch.randn(1, 224, 224, 3)
    tmp = Path("/tmp") / f"{stem}.export.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(tmp),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    out_dir = ROOT / "weights" / "classification" / stem
    write_pair_from_static(
        tmp,
        out_dir / f"{stem}.static_bs1.onnx",
        out_dir / f"{stem}.dyn.onnx",
    )


def main() -> int:
    export_one(models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1, "mobilenetv2")
    export_one(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, "resnet")
    export_one(models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "densenet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
