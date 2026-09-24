#!/usr/bin/env python3
"""Export product-matching EnlightenGAN ONNX from 200_net_G_A.pth.

Product C++ feeds NCHW input_src [1,3,H,W] and input_gray [1,1,H,W] already
normalized to [-1,1] / Rec.601 gray, H/W aligned to 16. The local enlighten.onnx
bakes 2x-1 + gray into a single `input` and is the wrong IO contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPSET, io_summary, sha256_of, write_pair_from_static  # noqa: E402

PTH = Path("/mnt/g/Codex/mortred_model_server/weights/enhancement/enlighten_gan/200_net_G_A.pth")
OUT_DIR = Path("/mnt/g/Codex/mortred_model_server/weights/enhancement/enlighten_gan")
CACHE = Path("/tmp/mortred_p2")
DUMMY_H, DUMMY_W = 256, 256


class UnetResizeConv(nn.Module):
    """VITA-Group EnlightenGAN Unet_resize_conv with self-attention + skip.

    pad_tensor / width>2200 pooling are omitted: C++ already align-up-16.
    """

    def __init__(self) -> None:
        super().__init__()
        p = 1
        self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = nn.BatchNorm2d(256)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv10 = nn.Conv2d(32, 3, 1)

    def _up(self, x: torch.Tensor) -> torch.Tensor:
        # pytorch_half_pixel / the existing enlighten.onnx Resize (opset 12).
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, inp: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        gray_2 = self.downsample_1(gray)
        gray_3 = self.downsample_2(gray_2)
        gray_4 = self.downsample_3(gray_3)
        gray_5 = self.downsample_4(gray_4)
        x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((inp, gray), 1))))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)
        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)
        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)
        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        x = self.max_pool4(conv4)
        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
        x = x * gray_5
        conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
        conv5 = self._up(conv5)
        conv4 = conv4 * gray_4
        up6 = torch.cat([self.deconv5(conv5), conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))
        conv6 = self._up(conv6)
        conv3 = conv3 * gray_3
        up7 = torch.cat([self.deconv6(conv6), conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))
        conv7 = self._up(conv7)
        conv2 = conv2 * gray_2
        up8 = torch.cat([self.deconv7(conv7), conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))
        conv8 = self._up(conv8)
        conv1 = conv1 * gray
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))
        latent = self.conv10(conv9) * gray
        return latent + inp


def _strip_prefix(state: dict) -> dict:
    if all(k.startswith("module.") for k in state):
        return {k[len("module.") :]: v for k, v in state.items()}
    if all(k.startswith("1.") for k in state):
        return {k[2:]: v for k, v in state.items()}
    return state


def main() -> int:
    try:
        state = torch.load(str(PTH), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(PTH), map_location="cpu")
    if not isinstance(state, dict) or not any(hasattr(v, "shape") for v in state.values()):
        raise SystemExit(f"unexpected pth payload: {type(state)}")
    state = _strip_prefix(state)
    net = UnetResizeConv()
    missing, unexpected = net.load_state_dict(state, strict=False)
    print("[enlighten] load missing", missing)
    print("[enlighten] load unexpected", unexpected)
    if missing:
        raise SystemExit("state_dict missing Unet weights")
    net.eval()
    dummy_src = torch.randn(1, 3, DUMMY_H, DUMMY_W)
    dummy_gray = torch.randn(1, 1, DUMMY_H, DUMMY_W)
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = CACHE / "enlightengan.export.onnx"
    torch.onnx.export(
        net,
        (dummy_src, dummy_gray),
        str(raw),
        input_names=["input_src", "input_gray"],
        output_names=["output"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_src": {0: "batch", 2: "height", 3: "width"},
            "input_gray": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "out_height", 3: "out_width"},
        },
    )
    static_dst = OUT_DIR / "enlightengan.static_bs1.onnx"
    dyn_dst = OUT_DIR / "enlightengan.dyn.onnx"
    write_pair_from_static(raw, static_dst, dyn_dst)
    # write_pair_from_static rewrites batch; keep spatial dynamic on both.
    print(f"[enlighten] static {io_summary(static_dst)}")
    print(f"[enlighten] dyn    {io_summary(dyn_dst)}")
    print(f"            sha static={sha256_of(static_dst)[:16]} dyn={sha256_of(dyn_dst)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
