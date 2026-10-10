import os
import sys
import joblib
import numpy as np

# Tambahkan root folder ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MODEL_PATH
from src.utils import extract_features

def predict_image(image_path):
    """
    Melakukan prediksi kesegaran buah berdasarkan path gambar.
    """
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model belum dilatih. Jalankan 'python -m src.train' terlebih dahulu."}
    
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return {"error": f"Gagal memuat model: {e}"}
    
    features = extract_features(image_path)
    if features is None:
        return {"error": "Gagal membaca atau memproses file gambar."}
    
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])
    confidence = float(np.max(probabilities))
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cara penggunaan: python -m src.predict <path_ke_gambar>")
    else:
        img_path = sys.argv[1]
        print(f"Menganalisis gambar: {img_path}...")
        result = predict_image(img_path)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n--- Hasil Analisis ---")
            print(f"Prediksi Kesegaran : {result['prediction'].upper()}")
            print(f"Tingkat Keyakinan  : {result['confidence'] * 100:.2f}%")