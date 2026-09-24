#!/usr/bin/env python3
"""P2.E OpenAI CLIP ViT-B/32 visual + text encoders (preprocess/tokenize stay in C++)."""
from __future__ import annotations

import sys
from pathlib import Path

import clip
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPSET, write_pair_from_static  # noqa: E402


class VisualEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(image)
        return feats / feats.norm(dim=-1, keepdim=True)


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)


def main() -> int:
    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval().float()
    out_dir = ROOT / "weights" / "openai_clip" / "vit-b-32"

    vis = VisualEncoder(model).eval()
    dummy_img = torch.randn(1, 3, 224, 224)
    tmp_v = Path("/tmp/mortred_p2/clip_visual.export.onnx")
    torch.onnx.export(
        vis,
        dummy_img,
        str(tmp_v),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    write_pair_from_static(tmp_v, out_dir / "visual.static_bs1.onnx", out_dir / "visual.dyn.onnx")

    txt = TextEncoder(model).eval()
    dummy_tok = torch.randint(0, 49408, (1, 77), dtype=torch.long)
    tmp_t = Path("/tmp/mortred_p2/clip_text.export.onnx")
    torch.onnx.export(
        txt,
        dummy_tok,
        str(tmp_t),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    write_pair_from_static(tmp_t, out_dir / "textual.static_bs1.onnx", out_dir / "textual.dyn.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
