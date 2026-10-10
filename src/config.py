import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Path penyimpanan model Random Forest
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_classifier.joblib")

# Resolusi standar gambar untuk ekstraksi fitur
IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42