# 🍎 Fruit Freshness Detector

Sistem pendeteksi kematangan buah berbasis **Deep Learning (CNN Transfer Learning MobileNetV2)**,
dilatih menggunakan dataset [Fruits Quality: Fresh vs Rotten](https://www.kaggle.com/datasets/nourabdoun/fruits-quality-fresh-vs-rotten) dari Kaggle.

---

## Kelas yang Didukung

| Buah | Fresh | Rotten |
|------|-------|--------|
| Apple | ✅ | ✅ |
| Banana | ✅ | ✅ |
| Bitter Gourd | ✅ | ✅ |
| Capsicum | ✅ | ✅ |
| Cucumber | ✅ | ✅ |
| Okra | ✅ | ✅ |
| Orange | ✅ | ✅ |
| Potato | ✅ | ✅ |
| Tomato | ✅ | ✅ |

---

## Struktur Proyek

```
fruit_freshness_detector/
├── data/
│   ├── train/          ← 70% data (dibuat oleh prepare_data.py)
│   ├── val/            ← 15% data
│   └── test/           ← 15% data
├── models/
│   ├── best_model.keras      ← model terbaik (dihasilkan training)
│   ├── final_model.keras     ← model akhir
│   ├── model_quantized.tflite← versi mobile (TFLite)
│   └── class_names.json      ← daftar nama kelas
├── outputs/
│   ├── training_log.csv
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── sample_predictions.png
│   └── classification_report.txt
├── src/
│   ├── config.py         ← semua hyperparameter & path
│   ├── prepare_data.py   ← split dataset ke train/val/test
│   ├── dataset.py        ← tf.data pipeline + augmentasi
│   ├── model.py          ← arsitektur CNN (MobileNetV2)
│   ├── train.py          ← script training utama
│   ├── evaluate.py       ← evaluasi mendalam + visualisasi
│   ├── predict.py        ← prediksi satu gambar / folder
│   └── utils.py          ← Grad-CAM, visualisasi, dll.
├── app/
│   └── app.py            ← aplikasi web Gradio
├── notebook_eksplorasi.ipynb
└── requirements.txt
```

---

## Cara Penggunaan

### 1. Instalasi

```bash
pip install -r requirements.txt
```

### 2. Download & Persiapkan Dataset

1. Download dataset dari [Kaggle](https://www.kaggle.com/datasets/nourabdoun/fruits-quality-fresh-vs-rotten)
2. Extract ke suatu folder, misal `~/datasets/fruits-quality`
3. Jalankan:

```bash
python src/prepare_data.py --src ~/datasets/fruits-quality --ratio 0.7 0.15 0.15
```

### 3. Training

```bash
# Training dengan parameter default
python src/train.py

# Kustomisasi parameter
python src/train.py --epochs 50 --batch_size 16 --lr 0.0005
```

Training berlangsung dua tahap:
- **Tahap 1** (Feature Extraction): backbone MobileNetV2 dibekukan
- **Tahap 2** (Fine-Tuning): sebagian backbone dibuka untuk penyesuaian

### 4. Evaluasi

```bash
python src/evaluate.py
```

Output: confusion matrix, per-class accuracy, classification report.

### 5. Prediksi

```bash
# Satu gambar
python src/predict.py --image path/gambar.jpg

# Seluruh folder
python src/predict.py --folder path/folder/ --output hasil.csv
```

### 6. Jalankan Aplikasi Web

```bash
python app/app.py
# Buka: http://localhost:7860

# Dengan link publik (Gradio share)
python app/app.py --share
```

---

## Arsitektur Model

```
Input (224×224×3)
    │
MobileNetV2 (ImageNet weights)
    │
Global Average Pooling
    │
BatchNormalization
    │
Dense(256, ReLU) + L2 reg
    │
Dropout(0.3)
    │
Dense(128, ReLU)
    │
Dropout(0.15)
    │
Dense(N_classes, Softmax)
```

---

## Hyperparameter Default

| Parameter | Nilai |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 30 |
| Learning rate (Tahap 1) | 1e-4 |
| Learning rate (Tahap 2) | 1e-5 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |

---

## Hasil yang Diharapkan

Model diharapkan mencapai akurasi **>90%** pada test set untuk klasifikasi per-kelas,
dan **>97%** untuk klasifikasi binary (Fresh vs Rotten).

---

*Dibuat untuk Praktikum / Tugas Akhir Teknik Komputer, Universitas Negeri Makassar*
