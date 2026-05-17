"""Export the trained YOLO11n model to ONNX (ready for tools.luxonis.com -> .blob).

Run AFTER training:
    python export.py
The ONNX lands at training/runs/yolo11n_banknote/weights/best.onnx.
Then upload best.pt OR best.onnx to https://tools.luxonis.com (RVC2, 6 shaves, 416x416).
"""
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BEST = HERE / "runs" / "yolo11n_banknote" / "weights" / "best.pt"


def main():
    assert BEST.exists(), f"missing {BEST} -- train first"
    model = YOLO(str(BEST))
    out = model.export(format="onnx", imgsz=416, opset=12, simplify=True)
    print("ONNX at:", out)


if __name__ == "__main__":
    main()
