"""
model.py
────────
Arsitektur model CNN berbasis Transfer Learning (MobileNetV2).

Strategi dua tahap:
  1. Feature extraction — backbone dibekukan, hanya head yang dilatih.
  2. Fine-tuning        — sebagian backbone dibuka untuk penyesuaian lebih lanjut.
"""

import tensorflow as tf
from src.config import (
    IMG_HEIGHT, IMG_WIDTH, CHANNELS,
    DROPOUT_RATE, DENSE_UNITS, FINE_TUNE_AT, LEARNING_RATE
)


def build_model(num_classes: int,
                learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Membangun model MobileNetV2 + custom classification head.

    Args:
        num_classes   : jumlah kelas output
        learning_rate : LR awal

    Returns:
        model yang sudah dikompilasi
    """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

    # ── Backbone ────────────────────────────────────────────
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False          # Tahap 1: bekukan semua layer

    # ── Head ────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               name="dense_1")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = tf.keras.layers.Dense(DENSE_UNITS // 2, activation="relu",
                               name="dense_2")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE / 2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="FruitFreshness_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    return model


def unfreeze_for_finetuning(model: tf.keras.Model,
                             fine_tune_at: int = FINE_TUNE_AT,
                             learning_rate: float = LEARNING_RATE * 0.1):
    """
    Membuka layer backbone mulai dari indeks `fine_tune_at` ke atas,
    lalu mengompilasi ulang dengan LR lebih kecil.
    """
    base_model = model.layers[1]           # layer ke-1 adalah MobileNetV2
    base_model.trainable = True

    # Bekukan layer sebelum fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\nFine-tune: {trainable_count} layer backbone dibuka "
          f"(mulai layer {fine_tune_at})")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


def model_summary(model: tf.keras.Model):
    model.summary(line_length=90)
    total   = model.count_params()
    train   = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\nTotal parameter : {total:,}")
    print(f"Trainable       : {train:,}")
    print(f"Non-trainable   : {total - train:,}")
