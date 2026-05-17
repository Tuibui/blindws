"""Convert CiRA Core banknote dataset -> Ultralytics YOLO format.

Reads:
    dataset_from_cira/datagen_cira/img/*.jpg + *.txt (Darknet labels with weird class IDs)
Writes:
    training/dataset/
        data.yaml
        images/train/*.jpg   (symlinks)
        images/val/*.jpg
        labels/train/*.txt   (remapped class IDs)
        labels/val/*.txt
"""
from __future__ import annotations
import os
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_IMG_DIR = REPO / "dataset_from_cira" / "datagen_cira" / "img"
DST = REPO / "training" / "dataset"

# CiRA Core internal class ID -> new YOLO class index (sorted by ID for determinism)
CLASS_MAP = {16: 0, 451: 1, 579: 2, 707: 3, 835: 4}
# Names line up with obj.names order; remap by guess (verify at inference)
CLASS_NAMES = ["20", "50", "100", "500", "1000"]

VAL_FRACTION = 0.10
SEED = 42


def main():
    assert SRC_IMG_DIR.is_dir(), f"missing {SRC_IMG_DIR}"
    images = sorted(SRC_IMG_DIR.glob("*.jpg"))
    assert images, "no jpg found"

    # Pair each image with its label, drop if either is missing
    pairs = []
    skipped = 0
    for img in images:
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            skipped += 1
            continue
        pairs.append((img, lbl))
    print(f"Found {len(pairs)} image/label pairs ({skipped} skipped)")

    # Shuffle + split
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Split: train={len(train_pairs)}  val={len(val_pairs)}")

    # Fresh output dirs
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    unknown_ids: set[int] = set()
    written = 0
    for split, items in (("train", train_pairs), ("val", val_pairs)):
        for img, lbl in items:
            # Symlink image (saves disk vs. copy; Ultralytics handles links fine)
            (DST / "images" / split / img.name).symlink_to(img.resolve())
            # Remap labels line by line
            out_lines = []
            for raw in lbl.read_text().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                cid = int(parts[0])
                if cid not in CLASS_MAP:
                    unknown_ids.add(cid)
                    continue
                parts[0] = str(CLASS_MAP[cid])
                out_lines.append(" ".join(parts))
            (DST / "labels" / split / lbl.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            written += 1

    if unknown_ids:
        print(f"WARNING: dropped {len(unknown_ids)} unknown class IDs: {sorted(unknown_ids)}")

    # data.yaml
    yaml = (
        f"path: {DST}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (DST / "data.yaml").write_text(yaml)
    print(f"Wrote {written} label files; data.yaml at {DST/'data.yaml'}")


if __name__ == "__main__":
    main()
