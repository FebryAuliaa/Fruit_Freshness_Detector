"""
evaluate.py
───────────
Evaluasi mendalam model: confusion matrix, classification report,
per-class accuracy, dan visualisasi prediksi.

Penggunaan:
  python src/evaluate.py
  python src/evaluate.py --model models/best_model.keras
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              ConfusionMatrixDisplay)

from src.config  import MODEL_DIR, OUTPUT_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
from src.dataset import load_dataset


def load_model_and_classes(model_path: str, class_file: str):
    model = tf.keras.models.load_model(model_path)
    with open(class_file) as f:
        class_names = json.load(f)
    return model, class_names


def get_predictions(model, dataset):
    y_true, y_pred_proba = [], []
    for images, labels in dataset:
        proba = model.predict(images, verbose=0)
        y_pred_proba.append(proba)
        y_true.append(labels.numpy())
    y_true       = np.argmax(np.vstack(y_true),       axis=1)
    y_pred_proba = np.vstack(y_pred_proba)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    return y_true, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    cm  = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1)

    # Warna hijau = fresh, merah = rotten
    colors = ["#2ecc71" if "fresh" in c.lower() else "#e74c3c"
              for c in class_names]

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.8), 5))
    bars = ax.bar(class_names, acc * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")

    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{a*100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Per-class accuracy → {path}")


def plot_sample_predictions(model, dataset, class_names, save_dir, n=16):
    """Tampilkan beberapa gambar beserta prediksi vs label sebenarnya."""
    images_all, labels_all = [], []
    for imgs, lbls in dataset.take(2):
        images_all.append(imgs.numpy())
        labels_all.append(lbls.numpy())
    images_all = np.concatenate(images_all)[:n]
    labels_all = np.concatenate(labels_all)[:n]

    proba    = model.predict(images_all, verbose=0)
    y_pred   = np.argmax(proba, axis=1)
    y_true   = np.argmax(labels_all, axis=1)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    # De-normalize (MobileNetV2 preprocess_input skala [-1,1] → [0,1])
    images_disp = (images_all + 1.0) / 2.0
    images_disp = np.clip(images_disp, 0, 1)

    for i in range(n):
        ax = axes[i]
        ax.imshow(images_disp[i])
        correct = y_pred[i] == y_true[i]
        color   = "green" if correct else "red"
        confidence = proba[i][y_pred[i]] * 100
        ax.set_title(
            f"Pred: {class_names[y_pred[i]]}\n"
            f"True: {class_names[y_true[i]]}  ({confidence:.1f}%)",
            color=color, fontsize=8
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Prediksi (Hijau=Benar, Merah=Salah)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Sample predictions → {path}")


def main(args):
    print("=" * 60)
    print("  Fruit Freshness Detector — Evaluasi")
    print("=" * 60)

    model, class_names = load_model_and_classes(
        args.model,
        os.path.join(MODEL_DIR, "class_names.json"),
    )

    test_ds, _ = load_dataset(TEST_DIR, subset="test", augment=False)

    print("\nMenghitung prediksi pada test set …")
    y_true, y_pred, y_pred_proba = get_predictions(model, test_ds)

    # ── Classification report ─────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4)
    print("\n" + report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report disimpan → {report_path}")

    # ── Visualisasi ───────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names, OUTPUT_DIR)
    plot_per_class_accuracy(y_true, y_pred, class_names, OUTPUT_DIR)
    plot_sample_predictions(model, test_ds, class_names, OUTPUT_DIR)

    # ── Ringkasan binary (fresh vs rotten) ───────────────────
    binary_true = np.array([0 if "fresh" in class_names[t].lower() else 1
                             for t in y_true])
    binary_pred = np.array([0 if "fresh" in class_names[p].lower() else 1
                             for p in y_pred])
    binary_report = classification_report(
        binary_true, binary_pred,
        target_names=["Fresh", "Rotten"],
        digits=4,
    )
    print("\nBinary (Fresh vs Rotten):")
    print(binary_report)

    print("\n✅ Evaluasi selesai. Semua output disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi model kematangan buah")
    parser.add_argument("--model",
                        default=os.path.join(MODEL_DIR, "best_model.keras"),
                        help="Path model .keras")
    args = parser.parse_args()
    main(args)
