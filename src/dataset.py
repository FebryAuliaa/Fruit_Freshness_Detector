"""
dataset.py
──────────
Membuat tf.data pipeline untuk train / val / test.
"""

import tensorflow as tf
from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, SEED, AUGMENTATION
)


# ── Augmentasi via keras layer (GPU-friendly) ────────────────
def build_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(AUGMENTATION["rotation_range"] / 360),
        tf.keras.layers.RandomZoom(AUGMENTATION["zoom_range"]),
        tf.keras.layers.RandomTranslation(
            height_factor=AUGMENTATION["height_shift_range"],
            width_factor=AUGMENTATION["width_shift_range"],
        ),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def _preprocess(image, label, augment: bool, aug_layer=None):
    image = tf.cast(image, tf.float32)
    # MobileNetV2 / EfficientNet gunakan skala [0,1] lewat preprocess_input
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    if augment and aug_layer is not None:
        image = aug_layer(image, training=True)
    return image, label


def load_dataset(directory: str,
                 subset: str = "training",
                 augment: bool = False) -> tf.data.Dataset:
    """
    Memuat gambar dari direktori menggunakan image_dataset_from_directory.
    Mengembalikan (dataset, class_names).
    """
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(subset == "training"),
        seed=SEED,
        interpolation="bilinear",
    )

    class_names = raw_ds.class_names

    aug_layer = build_augmentation_layer() if augment else None

    ds = raw_ds.map(
        lambda img, lbl: _preprocess(img, lbl, augment, aug_layer),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def get_datasets():
    """
    Mengembalikan (train_ds, val_ds, test_ds, class_names).
    """
    train_ds, class_names = load_dataset(TRAIN_DIR, subset="training",   augment=True)
    val_ds,   _           = load_dataset(VAL_DIR,   subset="validation", augment=False)
    test_ds,  _           = load_dataset(TEST_DIR,  subset="test",       augment=False)

    print(f"\nKelas ({len(class_names)}):")
    for i, c in enumerate(class_names):
        print(f"  [{i:2d}] {c}")

    return train_ds, val_ds, test_ds, class_names
