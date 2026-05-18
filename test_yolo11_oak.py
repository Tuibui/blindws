#!/usr/bin/env python3
"""Test the trained YOLO11n banknote detector on OAK-D-Lite (depthai 3.x).

Uses NN Archive (.tar.xz) + DetectionNetwork. Everything (preprocess, inference,
decode, NMS) runs on-chip; we just draw boxes.

NN Archive is produced by https://tools.luxonis.com (upload best.pt, target=RVC2,
size=416, YOLO11) OR locally via the `tools` CLI (luxonis/tools repo).

Controls: q to quit, +/- to adjust global confidence.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai

REPO = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = REPO / "trained_models" / "best.rvc2.tar.xz"

# Mapping is intrinsic to the model; DetectionNetwork.getClasses() returns names
# from the NN Archive, but we keep this human-friendly override here.
CLASS_NAMES_OVERRIDE = {0: "20", 1: "50", 2: "100", 3: "500", 4: "1000"}

INPUT_W = INPUT_H = 416   # must match the size you converted at on tools.luxonis.com

# Per-class confidence threshold (applied on host AFTER device NMS).
# Tune each index independently; +/- hotkeys shift them all by the same delta.
CONF_THRES_PER_CLASS = {
    0: 0.20,   # 20
    1: 0.20,   # 50
    2: 0.40,   # 100
    3: 0.20,   # 500
    4: 0.20,   # 1000
}
# Device-side threshold: keep low so all candidates reach the host for per-class filtering.
DEVICE_CONF_THRES = min(CONF_THRES_PER_CLASS.values())

# OAK-D-Lite is mounted upside down on your rig
ROTATE = dai.CameraImageOrientation.ROTATE_180_DEG
# Resize behaviour for preprocessing. CROP keeps aspect ratio + crops centre
# (matches CiRA dataset's square framing best). LETTERBOX adds gray bars;
# STRETCH distorts.
RESIZE_MODE = dai.ImgResizeMode.CROP


def passes(d, thres_by_class: dict[int, float]) -> bool:
    return d.confidence >= thres_by_class.get(int(d.label), 1.0)


def draw_detections(frame, detections, names, thres_by_class) -> None:
    H, W = frame.shape[:2]
    for d in detections:
        if not passes(d, thres_by_class):
            continue
        x1, y1 = int(d.xmin * W), int(d.ymin * H)
        x2, y2 = int(d.xmax * W), int(d.ymax * H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{names.get(d.label, str(d.label))} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE),
                    help="Path to NN Archive (.tar.xz) from tools.luxonis.com")
    ap.add_argument("--conf", type=float, default=DEVICE_CONF_THRES,
                    help="On-device confidence floor; per-class thresholds applied on host.")
    args = ap.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise SystemExit(
            f"missing {archive_path}\n"
            "Convert best.pt at https://tools.luxonis.com (RVC2, 416, YOLO11)\n"
            "and drop the .tar.xz here."
        )
    print(f"[i] archive: {archive_path}  ({archive_path.stat().st_size/1024:.1f} KB)")

    with dai.Pipeline() as pipe:
        cam = pipe.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.setImageOrientation(ROTATE)

        archive = dai.NNArchive(str(archive_path))
        det = pipe.create(dai.node.DetectionNetwork).build(
            cam, archive, fps=30, resizeMode=RESIZE_MODE,
        )
        det.setConfidenceThreshold(args.conf)

        names_from_archive = det.getClasses() or []
        # Prefer human-friendly override if it matches in length
        if names_from_archive and len(CLASS_NAMES_OVERRIDE) == len(names_from_archive):
            names = CLASS_NAMES_OVERRIDE
        else:
            names = {i: n for i, n in enumerate(names_from_archive)} or CLASS_NAMES_OVERRIDE
        print(f"[i] classes: {names}")
        print(f"[i] device conf floor: {args.conf}")
        print(f"[i] per-class thresholds: {CONF_THRES_PER_CLASS}")

        q_frame = det.passthrough.createOutputQueue(maxSize=4, blocking=False)
        q_det = det.out.createOutputQueue(maxSize=4, blocking=False)

        pipe.start()
        last = time.time()
        thres_by_class = dict(CONF_THRES_PER_CLASS)
        try:
            while pipe.isRunning():
                in_frame = q_frame.get()
                in_det = q_det.get()
                frame = in_frame.getCvFrame()
                detections = in_det.detections

                shown = [d for d in detections if passes(d, thres_by_class)]
                if shown:
                    summary = ", ".join(
                        f"id={d.label}({names.get(d.label, '?')}) s={d.confidence:.2f}"
                        for d in shown
                    )
                    print(f"[det] {len(shown)} -> {summary}")

                draw_detections(frame, detections, names, thres_by_class)

                now = time.time()
                fps = 1.0 / max(1e-6, now - last); last = now
                cv2.putText(frame, f"{fps:5.1f} FPS  {len(shown)} det",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("YOLO11n banknote", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("+") or key == ord("="):
                    thres_by_class = {k: min(0.99, v + 0.05) for k, v in thres_by_class.items()}
                    print(f"[thres] {thres_by_class}")
                elif key == ord("-") or key == ord("_"):
                    thres_by_class = {k: max(0.05, v - 0.05) for k, v in thres_by_class.items()}
                    print(f"[thres] {thres_by_class}")
        finally:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
