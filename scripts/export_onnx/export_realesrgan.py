#!/usr/bin/env python3
"""P2.F Real-ESRGAN x4v3 from official pth, NHWC wrapper (C++ sends RGB /255 NHWC)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPSET, write_pair_from_static  # noqa: E402


class SRVGGNetCompact(nn.Module):
    """Copied architecture from xinntao/Real-ESRGAN srvgg_arch.py (BSD-3-Clause)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        else:
            activation = nn.ReLU(inplace=True)
        self.body.append(activation)
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == "prelu":
                self.body.append(nn.PReLU(num_parameters=num_feat))
            else:
                self.body.append(nn.ReLU(inplace=True))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class NhwcInNchwOut(nn.Module):
    """C++ sends RGB /255 NHWC; postprocess still reads NCHW [1,3,H,W]."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 3, 1, 2).contiguous())


def main() -> int:
    ckpt = Path("/tmp/mortred_p2/realesr-general-x4v3.pth")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    params = state.get("params", state)
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    net.load_state_dict(params, strict=True)
    net.eval()
    wrapped = NhwcInNchwOut(net).eval()
    dummy = torch.rand(1, 64, 64, 3)
    tmp = Path("/tmp/mortred_p2/realesr-general-x4v3.export.onnx")
    torch.onnx.export(
        wrapped,
        dummy,
        str(tmp),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input": {1: "height", 2: "width"},
            "output": {2: "height_x4", 3: "width_x4"},
        },
    )
    out_dir = ROOT / "weights" / "enhancement" / "real_esrgan"
    write_pair_from_static(
        tmp,
        out_dir / "realesr-general-x4v3.static_bs1.onnx",
        out_dir / "realesr-general-x4v3.dyn.onnx",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
