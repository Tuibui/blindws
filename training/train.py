"""Train YOLO11n on the banknote dataset.

Target machine: RTX 5060 Ti 16GB -- device=0, batch=64 fits comfortably.
CPU fallback: set device="cpu", batch=4 (slow: ~6-12 hr).
"""
import os
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "data.yaml"

# Override from env if needed: DEVICE=cpu python train.py
DEVICE = os.environ.get("DEVICE", "0")
BATCH = int(os.environ.get("BATCH", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))


def main():
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} -- run `python prepare_dataset.py` first")
    model = YOLO("yolo11n.pt")  # auto-downloads pretrained COCO weights
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=416,
        batch=BATCH,
        device=DEVICE,
        project=str(HERE / "runs"),
        name="yolo11n_banknote",
        patience=30,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
