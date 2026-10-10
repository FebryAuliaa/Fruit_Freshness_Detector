"""
prepare_data.py
───────────────
Mempersiapkan dataset dari folder Kaggle ke struktur train/val/test.

Struktur folder Kaggle (setelah diextract):
  dataset/
    fresh_apple/
    fresh_banana/
    fresh_bitter_gourd/
    ...
    rotten_apple/
    rotten_banana/
    ...

Jalankan:
  python src/prepare_data.py --src /path/ke/dataset --ratio 0.7 0.15 0.15
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, SEED


def split_dataset(src_dir: str,
                  train_ratio: float = 0.70,
                  val_ratio:   float = 0.15,
                  test_ratio:  float = 0.15,
                  seed: int = SEED):
    """
    Membagi gambar dari src_dir ke train/val/test.
    Setiap subfolder di src_dir dianggap satu kelas.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Rasio harus berjumlah 1.0"

    random.seed(seed)
    src_path = Path(src_dir)

    classes = [d for d in src_path.iterdir() if d.is_dir()]
    print(f"\nDitemukan {len(classes)} kelas:")
    for c in sorted(classes):
        files = list(c.glob("*.*"))
        print(f"  {c.name:35s} → {len(files):5d} gambar")

    for cls_dir in tqdm(sorted(classes), desc="Split kelas"):
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR:   images[n_train : n_train + n_val],
            TEST_DIR:  images[n_train + n_val:],
        }

        for dest_root, imgs in splits.items():
            dest_cls = Path(dest_root) / cls_dir.name
            dest_cls.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest_cls / img.name)

    # Ringkasan
    print("\n✅ Split selesai:")
    for split_name, split_dir in [("Train", TRAIN_DIR),
                                   ("Val",   VAL_DIR),
                                   ("Test",  TEST_DIR)]:
        total = sum(len(list(Path(split_dir, c).glob("*.*")))
                    for c in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, c)))
        print(f"  {split_name:6s}: {total:6d} gambar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persiapan dataset buah")
    parser.add_argument("--src",   required=True, help="Folder sumber dataset Kaggle")
    parser.add_argument("--ratio", nargs=3, type=float,
                        default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Rasio train val test (default: 0.7 0.15 0.15)")
    args = parser.parse_args()

    split_dataset(args.src, *args.ratio)
