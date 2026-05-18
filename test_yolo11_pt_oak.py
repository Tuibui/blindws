#!/usr/bin/env python3
"""Test best.pt directly on OAK-D-Lite RGB feed (bypasses blob entirely).

If THIS detects but test_yolo11_oak.py (blob) does not -> conversion is broken.
If NEITHER detects on real banknotes -> dataset doesn't generalize, need more data.

Controls: q to quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent
DEFAULT_PT = REPO / "trained_models" / "best.pt"
CONF = 0.05
IMGSZ = 416


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=str(DEFAULT_PT))
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    pt = Path(args.pt)
    if not pt.exists():
        raise SystemExit(f"missing {pt}")
    print(f"[i] loading {pt}")
    model = YOLO(str(pt))
    print(f"[i] names: {model.names}")

    # OAK at native 1080p; Ultralytics handles letterbox + normalization itself
    pipe = dai.Pipeline()
    cam = pipe.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    out = cam.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.BGR888p,   # Ultralytics expects BGR (it does the swap internally)
        fps=30,
    )
    q = out.createOutputQueue(maxSize=4, blocking=False)

    pipe.start()
    last = time.time()
    try:
        while pipe.isRunning():
            frame = q.get().getCvFrame()
            # Ultralytics handles all preprocessing (letterbox, normalize, RGB swap)
            results = model.predict(frame, imgsz=IMGSZ, conf=args.conf, verbose=False)
            annotated = results[0].plot()

            now = time.time()
            fps = 1.0 / max(1e-6, now - last); last = now
            n = len(results[0].boxes)
            cv2.putText(annotated, f"{fps:5.1f} FPS  {n} det", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("best.pt on OAK RGB", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe.stop()
        try: pipe.wait()
        except Exception: pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
