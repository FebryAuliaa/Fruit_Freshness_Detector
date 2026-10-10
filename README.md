# 🍎 Fruit Freshness AI Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![GUI](https://img.shields.io/badge/GUI-Tkinter%20DnD-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit%20Learn-orange.svg)

Sistem Klasifikasi dan Deteksi Tingkat Kesegaran Buah Berbasis *Computer Vision* dan *Machine Learning* dengan Antarmuka Desktop Modern (*Drag & Drop*).

</div>

---

## 📌 Deskripsi Singkat Proyek

**Fruit Freshness AI Detector** adalah aplikasi cerdas berbasis desktop yang dirancang untuk mendeteksi tingkat kesegaran berbagai jenis buah secara otomatis menggunakan pendekatan *Computer Vision* dan algoritma *Machine Learning*. 

Proyek ini memproses dan mengekstraksi fitur visual dari gambar buah di dalam direktori dataset[cite: 2] (seperti warna, tekstur, atau histogram) untuk mengklasifikasikan apakah buah tersebut berada dalam kondisi segar atau tidak. Dilengkapi dengan antarmuka desktop bergaya *Modern Cyberpunk Dashboard*, aplikasi ini mendukung fitur **Drag & Drop** sehingga pengguna dapat dengan mudah menyeret gambar buah ke dalam aplikasi dan langsung melihat hasil analisis neural beserta tingkat keyakinan (*confidence score*) secara *real-time*.

---

## 🛠️ Fitur Utama

* **📁 Drag & Drop Interface**: Mempermudah pengguna memasukkan file gambar buah langsung ke aplikasi melalui gestur seret-dan-lepas (*drag and drop*).
* **🖼️ Live Image Preview**: Menampilkan pratinjau *thumbnail* gambar buah secara langsung di dalam antarmuka sebelum dianalisis oleh model.
* **📊 Analisis Neural Berbasis AI**: Memanfaatkan model *Machine Learning* yang telah dilatih (`joblib`) untuk mengekstrak fitur dan melakukan prediksi secara akurat.
* **📈 Dynamic Progress Bar**: Dilengkapi dengan indikator visual persentase tingkat keyakinan (*confidence*) yang dinamis.
* **🎨 Modern Dark UI**: Desain antarmuka profesional dengan palet warna gelap yang estetis dan nyaman dipandang.

---

## 📂 Struktur Direktori Proyek

```text
fruit_freshness_detector/
│
├── data/                  # Direktori dataset penyimpanan file gambar buah[cite: 2]
├── src/                   # Modul kode sumber utama
│   ├── config.py          # Konfigurasi path direktori dan parameter model
│   ├── utils.py           # Fungsi ekstraksi fitur gambar (Computer Vision)
│   └── train.py           # Skrip pelatihan model Machine Learning
├── app.py                 # Skrip antarmuka aplikasi desktop (GUI Modern & Drag & Drop)
└── README.md              # Dokumentasi lengkap proyek
