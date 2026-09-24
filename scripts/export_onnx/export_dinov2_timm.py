#!/usr/bin/env python3
"""DINOv2 via timm (official LVD-142M weights), cls token, 224 NCHW."""
from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

ROOT = Path("/mnt/g/Codex/mortred_model_server")
sys.path.insert(0, str(ROOT / "scripts" / "export_onnx"))
from common import OPSET, write_pair_from_static  # noqa: E402

MODELS = {
    "vits14": "vit_small_patch14_dinov2.lvd142m",
    "vitb14": "vit_base_patch14_dinov2.lvd142m",
    "vitl14": "vit_large_patch14_dinov2.lvd142m",
}


class ClsOnly(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_one(key: str, timm_name: str) -> None:
    print(f"=== DINOv2 {key} {timm_name} ===", flush=True)
    net = timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=224)
    net.eval()
    wrapped = ClsOnly(net).eval()
    dummy = torch.randn(1, 3, 224, 224)
    tmp = Path("/tmp/mortred_p2") / f"dinov2_{key}.export.onnx"
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
    out_dir = ROOT / "weights" / "classification" / "dinov2"
    write_pair_from_static(
        tmp,
        out_dir / f"dinov2_{key}_pretrain.static_bs1.onnx",
        out_dir / f"dinov2_{key}_pretrain.dyn.onnx",
    )


def main() -> int:
    keys = sys.argv[1:] or ["vits14"]
    for key in keys:
        export_one(key, MODELS[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
