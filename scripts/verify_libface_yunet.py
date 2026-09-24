#!/usr/bin/env python3
"""WSL check for LibFace YuNet decode vs OpenCV FaceDetectorYN.

Mirrors src/models/object_detection/libface_detector.inl:
  BGR uint8 -> optional DIRECT_RESIZE -> right/bottom pad to /32 -> float NCHW
  score = sqrt(clamp(cls)*clamp(obj)); bbox/kps decode at strides 8/16/32
  NMS via cv2.dnn.NMSBoxes (same helper the C++ tail uses)

Examples (from repo root, inside WSL):

  python3 scripts/verify_libface_yunet.py
  python3 scripts/verify_libface_yunet.py --size 320x240 --size 640x480
  python3 scripts/verify_libface_yunet.py --native

C++ golden (after a WSL cmake build), still uses 640x480_config.toml:

  MORTRED_UPDATE_GOLDEN=1 ./build-cpu/bin/model_golden_test \\
      --gtest_filter=model_golden.libface_detection
  ./build-cpu/bin/model_golden_test --gtest_filter=model_golden.libface_detection
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "weights/object_detection/libfacedetection/face_detection_yunet_2026may.onnx"
DEFAULT_IMAGE = ROOT / "demo_data/model_test_input/object_detection/face_wo_mask.jpg"
STRIDES = (8, 16, 32)
HEADS = ("cls", "obj", "bbox", "kps")


def parse_size(text: str) -> tuple[int, int]:
    height, width = text.lower().split("x")
    return int(width), int(height)


def align_hw(width: int, height: int, divisor: int = 32) -> tuple[int, int]:
    return ((width + divisor - 1) // divisor) * divisor, ((height + divisor - 1) // divisor) * divisor


def preprocess(bgr: np.ndarray, size_wh: tuple[int, int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    working = bgr
    if size_wh is not None:
        width, height = size_wh
        working = cv2.resize(working, (width, height), interpolation=cv2.INTER_LINEAR)
    unpadded = (working.shape[1], working.shape[0])
    pad_w, pad_h = align_hw(*unpadded)
    if (pad_w, pad_h) != unpadded:
        working = cv2.copyMakeBorder(working, 0, pad_h - unpadded[1], 0, pad_w - unpadded[0], cv2.BORDER_CONSTANT, value=0)
    blob = working.astype(np.float32).transpose(2, 0, 1)[None, ...]
    return blob, unpadded


def decode(outputs: dict[str, np.ndarray], pad_wh: tuple[int, int], score_threshold: float) -> list[dict]:
    pad_w, pad_h = pad_wh
    faces: list[dict] = []
    for stride in STRIDES:
        cls = np.asarray(outputs[f"cls_{stride}"], dtype=np.float32).reshape(-1)
        obj = np.asarray(outputs[f"obj_{stride}"], dtype=np.float32).reshape(-1)
        bbox = np.asarray(outputs[f"bbox_{stride}"], dtype=np.float32).reshape(-1, 4)
        kps = np.asarray(outputs[f"kps_{stride}"], dtype=np.float32).reshape(-1, 10)
        cols = pad_w // stride
        rows = pad_h // stride
        expected = rows * cols
        if cls.shape[0] != expected or bbox.shape[0] != expected:
            raise RuntimeError(f"stride {stride}: got {cls.shape[0]} anchors, expected {expected} ({rows}x{cols})")
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                score = math.sqrt(float(np.clip(cls[idx], 0.0, 1.0) * np.clip(obj[idx], 0.0, 1.0)))
                if score < score_threshold:
                    continue
                cx = (col + float(bbox[idx, 0])) * stride
                cy = (row + float(bbox[idx, 1])) * stride
                width = math.exp(float(bbox[idx, 2])) * stride
                height = math.exp(float(bbox[idx, 3])) * stride
                landmarks = []
                for n in range(5):
                    px = (float(kps[idx, 2 * n]) + col) * stride
                    py = (float(kps[idx, 2 * n + 1]) + row) * stride
                    landmarks.append((px, py))
                faces.append(
                    {
                        "x": cx - width * 0.5,
                        "y": cy - height * 0.5,
                        "w": width,
                        "h": height,
                        "score": score,
                        "landmarks": landmarks,
                    }
                )
    return faces


def nms(faces: list[dict], score_threshold: float, nms_threshold: float, keep_top_k: int) -> list[dict]:
    if not faces:
        return []
    boxes = [[f["x"], f["y"], f["w"], f["h"]] for f in faces]
    scores = [float(f["score"]) for f in faces]
    kept = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    if kept is None or len(kept) == 0:
        return []
    indices = [int(i) for i in np.array(kept).reshape(-1)]
    result = [faces[i] for i in indices]
    if len(result) > keep_top_k:
        result = result[:keep_top_k]
    return result


def scale_to_source(faces: list[dict], unpadded: tuple[int, int], source_wh: tuple[int, int]) -> list[dict]:
    sx = source_wh[0] / float(unpadded[0])
    sy = source_wh[1] / float(unpadded[1])
    out = []
    for face in faces:
        mapped = dict(face)
        mapped["x"] = face["x"] * sx
        mapped["y"] = face["y"] * sy
        mapped["w"] = face["w"] * sx
        mapped["h"] = face["h"] * sy
        mapped["landmarks"] = [(p[0] * sx, p[1] * sy) for p in face["landmarks"]]
        out.append(mapped)
    return out


def iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def print_faces(title: str, faces: list[dict]) -> None:
    print(f"{title}: {len(faces)} faces")
    for i, face in enumerate(faces):
        print(
            f"  [{i}] score={face['score']:.4f} "
            f"xywh=({face['x']:.1f},{face['y']:.1f},{face['w']:.1f},{face['h']:.1f})"
        )


def opencv_yunet(onnx: Path, image: np.ndarray, score: float, nms_thr: float, top_k: int) -> list[dict] | None:
    major = int(cv2.__version__.split(".", 1)[0])
    if major < 5 or not hasattr(cv2, "FaceDetectorYN"):
        return None
    h, w = image.shape[:2]
    try:
        detector = cv2.FaceDetectorYN.create(str(onnx), "", (w, h), score, nms_thr, top_k)
        detector.setInputSize((w, h))
        _retval, faces = detector.detect(image)
    except cv2.error:
        return None
    if faces is None:
        return []
    out = []
    for row in np.asarray(faces):
        out.append(
            {
                "x": float(row[0]),
                "y": float(row[1]),
                "w": float(row[2]),
                "h": float(row[3]),
                "score": float(row[14]),
                "landmarks": [(float(row[4 + 2 * n]), float(row[5 + 2 * n])) for n in range(5)],
            }
        )
    return out


def match_report(ours: list[dict], ref: list[dict], iou_thresh: float = 0.5) -> tuple[int, float]:
    used = [False] * len(ref)
    matched = 0
    worst = 1.0
    for face in ours:
        best_i, best = -1, 0.0
        for i, other in enumerate(ref):
            if used[i]:
                continue
            value = iou(face, other)
            if value > best:
                best, best_i = value, i
        if best_i >= 0 and best >= iou_thresh:
            used[best_i] = True
            matched += 1
            worst = min(worst, best)
    return matched, (0.0 if matched == 0 else worst)


def run_ort(onnx: Path, blob: np.ndarray) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_outputs()]
    got = session.run(names, {"input": blob})
    return {name: value for name, value in zip(names, got)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--size", action="append", default=[], help="HxW, repeatable. Default: 320x240 and 640x480")
    parser.add_argument("--native", action="store_true", help="also run at the source resolution (OpenCV demo path)")
    parser.add_argument("--score", type=float, default=0.75)
    parser.add_argument("--nms", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=250)
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"missing onnx: {args.onnx}", file=sys.stderr)
        return 1
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"failed to read {args.image}", file=sys.stderr)
        return 1

    jobs: list[tuple[str, tuple[int, int] | None]] = []
    sizes = args.size or ["320x240", "640x480"]
    for item in sizes:
        width, height = parse_size(item)
        jobs.append((item, (width, height)))
    if args.native:
        jobs.append(("native", None))

    source_wh = (image.shape[1], image.shape[0])
    print(f"onnx={args.onnx}")
    print(f"image={args.image} {source_wh[0]}x{source_wh[1]}")
    print(f"thresholds score={args.score} nms={args.nms} top_k={args.top_k}")

    failed = 0
    for label, size_wh in jobs:
        blob, unpadded = preprocess(image, size_wh)
        pad_wh = (blob.shape[3], blob.shape[2])
        outputs = run_ort(args.onnx, blob)
        for stride in STRIDES:
            cls = outputs[f"cls_{stride}"]
            print(f"  cls_{stride}{list(cls.shape)} bbox_{stride}{list(outputs[f'bbox_{stride}'].shape)}")
        raw = decode(outputs, pad_wh, args.score)
        kept = nms(raw, args.score, args.nms, args.top_k)
        mapped = scale_to_source(kept, unpadded, source_wh)
        print(f"\n== {label} unpadded={unpadded[0]}x{unpadded[1]} padded={pad_wh[0]}x{pad_wh[1]} ==")
        print_faces("ORT+C++-decode (source image coords)", mapped)

        resized = image if size_wh is None else cv2.resize(image, size_wh, interpolation=cv2.INTER_LINEAR)
        ref = opencv_yunet(args.onnx, resized, args.score, args.nms, args.top_k)
        if ref is None:
            print("OpenCV FaceDetectorYN skipped (2026may needs OpenCV 5.x; boxes above are ORT + the C++ decoder)")
            continue
        ref_src = scale_to_source(ref, (resized.shape[1], resized.shape[0]), source_wh)
        print_faces("OpenCV FaceDetectorYN (source image coords)", ref_src)
        matched, worst = match_report(mapped, ref_src)
        print(f"match {matched}/{len(ref_src)} (ours {len(mapped)}) worst_iou={worst:.4f}")
        if len(mapped) != len(ref_src) or matched != len(ref_src):
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
