# Banknote YOLO11n training

Retraining banknote detector with Ultralytics YOLO11n because CiRA Core's `.weights`
is encrypted (scytale cipher) and unloadable by any standard converter.

## Files in this repo

| Path | Tracked | Notes |
|---|---|---|
| `dataset_from_cira/datagen_cira/img/` | ✅ | 1672 jpg + Darknet `.txt` labels (plain) |
| `dataset_from_cira/datagen_cira/{obj.*,*.cfg,train_darknet.txt}` | ✅ | Originals from CiRA DeepTrain |
| `dataset_from_cira/datagen_cira/backup/` | ❌ | Encrypted weights, useless |
| `training/prepare_dataset.py` | ✅ | Builds Ultralytics dataset (remap class IDs + split) |
| `training/train.py` | ✅ | YOLO11n training entry |
| `training/export.py` | ✅ | `.pt` → `.onnx` |
| `training/dataset/` | ❌ | Generated, do not edit |
| `training/runs/` | ❌ | Training output |

## Class mapping (guess — verify at inference)

| YOLO id | label (baht) | CiRA Core internal id |
|---|---|---|
| 0 | 20 | 16 |
| 1 | 50 | 451 |
| 2 | 100 | 579 |
| 3 | 500 | 707 |
| 4 | 1000 | 835 |

If a class appears wrong at inference, add a remap dict in the consuming code.

## Training on the GPU machine (RTX 5060 Ti 16 GB)

```bash
# 1) Clone + venv
git clone https://github.com/Tuibui/blindws.git
cd blindws
python3 -m venv venv && source venv/bin/activate

# 2) Install torch matching your CUDA (5060 Ti = sm_120, needs CUDA 12.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) Install the rest
pip install -r training/requirements.txt

# 4) Build the Ultralytics dataset (one-shot)
python training/prepare_dataset.py

# 5) Verify GPU
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 6) Train -- ~15-30 min on RTX 5060 Ti
python training/train.py
# override knobs if needed:  EPOCHS=200 BATCH=128 python training/train.py

# 7) Export ONNX
python training/export.py
# best.pt + best.onnx land in training/runs/yolo11n_banknote/weights/

# 8) Upload best.pt OR best.onnx to https://tools.luxonis.com
#    target: RVC2  |  shaves: 6  |  input: 416x416
#    drop the resulting .blob into trained_models/
```

## Why the CiRA `.weights` cannot be used

`dataset_from_cira/datagen_cira/backup/train.backup` starts with bytes:

```
00000000: 0000 0000 0200 0000 0500 0000 ...    (standard Darknet 20-byte header)
00000010: 0000 0000 5f3c 7363 7974 616c 653e 5f00   ..._<scytale>_.
```

That `_<scytale>_` marker indicates CiRA Core scrambles all weight bytes after the header.
Both Tianxiaomo's PyTorch loader and OpenCV's C++ Darknet loader produce NaN output —
the bytes where BN `running_var` would be parse as negative values, which is impossible
for a real variance. Same signature appears in another public CiRA Core export
(`tonpai1007/CiraCore`), confirming this is a deliberate DRM, not a bug.
