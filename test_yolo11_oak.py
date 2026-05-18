#!/usr/bin/env python3
"""Test the trained YOLO11n banknote detector on OAK-D-Lite (depthai 3.x).

Host-side post-processing (no Luxonis JSON config needed).
Controls: q to quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_BLOB = REPO / "trained_models" / "best.blob"

# YOLO id -> human label (guessed at training; override once verified by eye)
CLASS_NAMES = {0: "20", 1: "50", 2: "100", 3: "500", 4: "1000"}
NUM_CLASSES = len(CLASS_NAMES)

INPUT_W = INPUT_H = 416   # matches training/train.py imgsz=416

# Per-class confidence threshold. Tune each index independently.
CONF_THRES_PER_CLASS = {
    0: 0.50,   # 20
    1: 0.80,   # 50
    2: 0.3,   # 100
    3: 0.6,   # 500
    4: 0.60,   # 1000
}
# Vectorised lookup; default to 1.0 (reject) for any unknown class id
CONF_THRES_VEC = np.array(
    [CONF_THRES_PER_CLASS.get(i, 1.0) for i in range(len(CLASS_NAMES))],
    dtype=np.float32,
)
IOU_THRES = 0.05


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thres]
    return keep


def decode_yolo11(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw YOLO11 head -> (boxes_xyxy_norm, scores, class_ids).

    Ultralytics YOLO11 output is (1, 4+nc, N) -- 4 bbox channels in pixels +
    nc class scores (already sigmoid). N = sum of grid sizes (3549 for 416 input).
    Some exports swap to (1, N, 4+nc), so we sniff the layout.
    """
    if out.ndim == 3 and out.shape[1] == 4 + NUM_CLASSES:
        data = out[0]                       # (4+nc, N)
    elif out.ndim == 3 and out.shape[2] == 4 + NUM_CLASSES:
        data = out[0].T                     # (4+nc, N)
    else:
        raise RuntimeError(f"unexpected YOLO11 output shape: {out.shape}")

    cx, cy, w, h = data[0], data[1], data[2], data[3]
    cls_scores = data[4:4 + NUM_CLASSES]
    class_ids = cls_scores.argmax(axis=0)
    scores = cls_scores.max(axis=0)

    # Per-class threshold: each anchor must beat its own winning class's threshold
    mask = scores >= CONF_THRES_VEC[class_ids]
    if not mask.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    x1 = (cx - w / 2) / INPUT_W
    y1 = (cy - h / 2) / INPUT_H
    x2 = (cx + w / 2) / INPUT_W
    y2 = (cy + h / 2) / INPUT_H
    boxes = np.clip(np.stack([x1, y1, x2, y2], axis=1), 0.0, 1.0)
    return boxes, scores, class_ids


def draw_detections(frame, boxes, scores, class_ids) -> None:
    H, W = frame.shape[:2]
    for (x1n, y1n, x2n, y2n), s, cid in zip(boxes, scores, class_ids):
        x1, y1 = int(x1n * W), int(y1n * H)
        x2, y2 = int(x2n * W), int(y2n * H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASS_NAMES.get(int(cid), str(int(cid)))} {s:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default=str(DEFAULT_BLOB))
    args = ap.parse_args()

    blob = Path(args.blob)
    if not blob.exists():
        raise SystemExit(f"missing blob: {blob}")
    print(f"[i] blob: {blob}  ({blob.stat().st_size/1024:.1f} KB)")

    pipe = dai.Pipeline()

    cam = pipe.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    preview = cam.requestOutput(
        size=(INPUT_W, INPUT_H),
        type=dai.ImgFrame.Type.RGB888p,   # YOLO trained on RGB; depthai default is BGR
        fps=30,
    )

    nn = pipe.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob))
    nn.setNumInferenceThreads(2)
    preview.link(nn.input)

    q_rgb = preview.createOutputQueue(maxSize=4, blocking=False)
    q_nn  = nn.out.createOutputQueue(maxSize=4, blocking=False)

    pipe.start()
    last = time.time()
    try:
        while pipe.isRunning():
            in_nn = q_nn.get()
            in_rgb = q_rgb.get()

            frame = in_rgb.getCvFrame()
            tensor = np.asarray(in_nn.getFirstTensor(), dtype=np.float32)
            # YOLO11n / 5 cls / 416 -> (1, 9, 3549) (or transposed). Add batch dim if missing.
            if tensor.ndim == 2:
                tensor = tensor[None, ...]
            try:
                if tensor.shape[1] == 4 + NUM_CLASSES or tensor.shape[2] == 4 + NUM_CLASSES:
                    out = tensor
                else:
                    out = tensor.reshape(1, 4 + NUM_CLASSES, -1)
            except Exception as e:
                print(f"reshape failed; tensor shape={tensor.shape} layers={in_nn.getAllLayerNames()}: {e}")
                continue

            boxes, scores, cids = decode_yolo11(out)
            if len(boxes):
                keep = nms(boxes, scores, IOU_THRES)
                boxes, scores, cids = boxes[keep], scores[keep], cids[keep]
                dets = ", ".join(
                    f"id={int(c)}({CLASS_NAMES.get(int(c), '?')}) s={s:.2f}"
                    for c, s in zip(cids, scores)
                )
                print(f"[det] {len(boxes)} -> {dets}")

            draw_detections(frame, boxes, scores, cids)
            now = time.time()
            fps = 1.0 / max(1e-6, now - last); last = now
            cv2.putText(frame, f"{fps:5.1f} FPS  {len(boxes)} det", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("YOLO11n banknote", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe.stop()
        try:
            pipe.wait()
        except Exception:
            pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
