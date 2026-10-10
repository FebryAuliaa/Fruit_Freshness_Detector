import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from src.config import DATA_DIR, MODEL_PATH, MODEL_DIR, RANDOM_STATE
from src.utils import extract_features

def train_model():
    print("Memulai proses ekstraksi fitur dari dataset...")
    X, y = [], []
    
    if not os.path.exists(DATA_DIR):
        print(f"Direktori data tidak ditemukan di: {DATA_DIR}")
        print("Pastikan struktur folder data memiliki subfolder berdasarkan label (misal: data/fresh dan data/rotten).")
        return

    # Iterasi folder berdasarkan nama kelas/label
    for label_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Memproses kelas: {label_name}...")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_name)
                
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("Error: Tidak ada data gambar yang berhasil diproses.")
        return

    # Membagi dataset menjadi data latih dan data uji (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    print("Melatih model Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    
    # Evaluasi Model
    y_pred = clf.predict(X_test)
    print(f"\nAkurasi Model: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    # Menyimpan Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model berhasil disimpan ke: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()