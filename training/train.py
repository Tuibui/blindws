"""Train YOLO11n on the banknote dataset -- aggressive aug for distance/scale invariance.

Target: RTX 5060 Ti 16GB. CPU fallback: DEVICE=cpu BATCH=4 python train.py.

Knobs you can change without touching code:
    EPOCHS=150 BATCH=64 IMGSZ=416 python train.py

Why strong aug? CiRA Core dataset was captured at narrow distance/angle range,
so the model overfits to that scale. Mosaic + mixup + copy_paste + wider scale
force the model to learn objects at many sizes and contexts.
"""
import os
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "data.yaml"

DEVICE = os.environ.get("DEVICE", "0")
BATCH = int(os.environ.get("BATCH", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))   # stronger aug -> needs a few more epochs
IMGSZ = int(os.environ.get("IMGSZ", "416"))      # bump to 640 if blob can be re-converted


def main():
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} -- run `python prepare_dataset.py` first")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(HERE / "runs"),
        name="yolo11n_banknote",
        patience=40,
        save=True,
        plots=True,
        # ---------- Geometric augmentation (the big distance/scale fix) ----------
        scale=0.9,          # 0.5 default -> 0.9: random scale 0.1x ~ 1.9x (covers near/far)
        degrees=15,         # 0 default   -> ±15 deg rotation (hand-held angles)
        translate=0.15,     # 0.1 default -> shift up to 15%
        shear=2.0,          # 0 default   -> mild shear
        perspective=0.0005, # 0 default   -> tiny perspective warp (camera angles)
        fliplr=0.5,         # default     -> horizontal flip OK for banknotes
        flipud=0.0,         # default     -> NO vertical flip (banknote orientation matters)
        # ---------- Photometric augmentation (lighting/color robustness) ----------
        hsv_h=0.02,         # 0.015 default -> slight hue jitter
        hsv_s=0.8,          # 0.7 default   -> stronger saturation jitter
        hsv_v=0.5,          # 0.4 default   -> brightness jitter (dim/bright scenes)
        # ---------- Composition augmentation (multi-scale & context) -------------
        mosaic=1.0,         # default       -> 4 imgs combined, simulates many scales
        close_mosaic=15,    # 10 default    -> turn mosaic OFF in last 15 epochs for clean fit
        mixup=0.15,         # 0 default     -> blend two images
        copy_paste=0.3,     # 0 default     -> paste objects across images (small-obj boost)
        erasing=0.4,        # 0.4 default   -> random erase (occlusion robustness)
        # ---------- Optimisation ---------------------------------------------------
        cos_lr=True,        # cosine LR schedule, plays nice with longer epochs
        warmup_epochs=3.0,
    )


if __name__ == "__main__":
    main()
