import cv2
import numpy as np
from src.config import IMAGE_SIZE

def extract_features(image_path):
    """
    Mengekstrak fitur numerik dari gambar menggunakan Color Histogram dan Canny Edge 
    menggunakan OpenCV murni tanpa memerlukan scikit-image atau compiler C++.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Ubah ukuran gambar agar seragam
    image = cv2.resize(image, IMAGE_SIZE)
    
    # 1. Ekstraksi Fitur Warna (HSV Color Histogram)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_features = hist.flatten()
    
    # 2. Ekstraksi Fitur Tekstur/Tepi menggunakan Grayscale & Canny
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(gray_hist, gray_hist)
    
    edges = cv2.Canny(gray, 100, 200)
    edge_mean = np.mean(edges)
    edge_std = np.std(edges)
    
    # Gabungkan semua fitur menjadi satu vektor numerik
    combined_features = np.hstack([hist_features, gray_hist.flatten(), [edge_mean, edge_std]])
    return combined_features