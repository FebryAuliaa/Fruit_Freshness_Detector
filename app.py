import os
import sys
import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox, Canvas
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

# Tambahkan root folder ke sys.path jika dijalankan dari dalam folder app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MODEL_PATH
from src.utils import extract_features

class UltraModernFruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FruitFresh AI - Intelligent Vision System")
        self.root.geometry("820x540")
        self.root.config(bg="#090D16")
        self.root.resizable(False, False)

        # Load Model
        self.model = self.load_model()
        self.current_image_path = None
        self.tk_image = None

        # --- TOP APP BAR ---
        top_bar = tk.Frame(root, bg="#0F1626", height=60)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)

        logo_label = tk.Label(
            top_bar, 
            text="⚡ FruitFresh Vision AI", 
            font=("Segoe UI", 12, "bold"), 
            bg="#0F1626", 
            fg="#00F5A0"
        )
        logo_label.pack(side=tk.LEFT, padx=20)

        version_label = tk.Label(
            top_bar, 
            text="v2.0 Neural Engine", 
            font=("Segoe UI", 9), 
            bg="#0F1626", 
            fg="#627D98"
        )
        version_label.pack(side=tk.RIGHT, padx=20)

        # --- MAIN WORKSPACE (SPLIT PANEL) ---
        workspace = tk.Frame(root, bg="#090D16")
        workspace.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # --- LEFT PANEL: INPUT & DRAG DROP ---
        left_panel = tk.Frame(workspace, bg="#0F1626", width=370)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)

        panel_title_left = tk.Label(
            left_panel, text="1. MASUKKAN GAMBAR BUAH", 
            font=("Segoe UI", 10, "bold"), bg="#0F1626", fg="#829AB1"
        )
        panel_title_left.pack(anchor="w", padx=20, pady=(20, 10))

        # Drop Zone Card
        self.drop_frame = tk.Frame(
            left_panel, 
            bg="#141E33", 
            highlightbackground="#243B53", 
            highlightthickness=2, 
            bd=0
        )
        self.drop_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        self.drop_frame.pack_propagate(False)

        self.drop_content = tk.Label(
            self.drop_frame, 
            text="📂\nDrag & Drop Gambar Di Sini\n(Format: JPG, PNG)", 
            font=("Segoe UI", 10), 
            bg="#141E33", 
            fg="#627D98",
            justify="center"
        )
        self.drop_content.pack(expand=True, fill=tk.BOTH)

        # Daftarkan Drag & Drop
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)

        # Tombol Aksi AI
        self.predict_btn = tk.Button(
            left_panel, 
            text="JALANKAN PREDIKSI AI", 
            font=("Segoe UI", 10, "bold"), 
            bg="#243B53", 
            fg="#829AB1", 
            bd=0, 
            cursor="arrow",
            state=tk.DISABLED, 
            command=self.predict_image,
            height=2,
            activebackground="#00F5A0",
            activeforeground="#090D16"
        )
        self.predict_btn.pack(fill=tk.X, padx=20, pady=(0, 20))

        # --- RIGHT PANEL: AI ANALYTICS RESULT ---
        right_panel = tk.Frame(workspace, bg="#0F1626", width=390)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        right_panel.pack_propagate(False)

        panel_title_right = tk.Label(
            right_panel, text="2. HASIL ANALISIS NEURAL", 
            font=("Segoe UI", 10, "bold"), bg="#0F1626", fg="#829AB1"
        )
        panel_title_right.pack(anchor="w", padx=20, pady=(20, 15))

        # Result Card Container
        result_inner = tk.Frame(right_panel, bg="#141E33", bd=0)
        result_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Status Kesegaran Label
        tk.Label(
            result_inner, text="STATUS KESEGARAN", 
            font=("Segoe UI", 8, "bold"), bg="#141E33", fg="#627D98"
        ).pack(anchor="w", padx=20, pady=(20, 2))

        self.status_value_label = tk.Label(
            result_inner, text="MENUNGGU INPUT...", 
            font=("Segoe UI", 16, "bold"), bg="#141E33", fg="#486581"
        )
        self.status_value_label.pack(anchor="w", padx=20, pady=(0, 15))

        # Separator line
        sep = tk.Frame(result_inner, bg="#243B53", height=1)
        sep.pack(fill=tk.X, padx=20, pady=5)

        # Confidence Score Label
        tk.Label(
            result_inner, text="TINGKAT KEYAKINAN (CONFIDENCE)", 
            font=("Segoe UI", 8, "bold"), bg="#141E33", fg="#627D98"
        ).pack(anchor="w", padx=20, pady=(15, 5))

        self.conf_value_label = tk.Label(
            result_inner, text="-%", 
            font=("Segoe UI", 14, "bold"), bg="#141E33", fg="#F0F4F8"
        )
        self.conf_value_label.pack(anchor="w", padx=20, pady=(0, 10))

        # Custom Progress Bar (Canvas)
        self.canvas_bg_color = "#1E293B"
        self.canvas_fill_color = "#00F5A0"
        
        self.progress_canvas = Canvas(
            result_inner, width=310, height=10, bg=self.canvas_bg_color, highlightthickness=0
        )
        self.progress_canvas.pack(padx=20, pady=(0, 25))
        self.progress_bar = self.progress_canvas.create_rectangle(0, 0, 0, 10, fill=self.canvas_fill_color, width=0)

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        return None

    def handle_drop(self, event):
        file_path = event.data.strip('{}')
        self.process_image(file_path)

    def process_image(self, file_path):
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            messagebox.showerror("Error", "Harap masukkan file gambar berformat .jpg, .jpeg, atau .png!")
            return

        self.current_image_path = file_path
        
        try:
            img = Image.open(file_path)
            img.thumbnail((160, 130))
            self.tk_image = ImageTk.PhotoImage(img)

            self.drop_content.config(text="", image=self.tk_image, bg="#141E33")
            self.predict_btn.config(state=tk.NORMAL, bg="#00F5A0", fg="#090D16", cursor="hand2")
            
            self.status_value_label.config(text="SIAP DIANALISIS", fg="#38BDF8")
            self.conf_value_label.config(text="0.00%")
            self.update_progress_bar(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca gambar: {e}")

    def update_progress_bar(self, percentage):
        width = (percentage / 100) * 310
        self.progress_canvas.coords(self.progress_bar, 0, 0, width, 10)

    def predict_image(self):
        if not self.current_image_path:
            return

        if self.model is None:
            messagebox.showerror("Error", "Model belum dilatih. Jalankan 'python -m src.train' terlebih dahulu.")
            return

        try:
            self.predict_btn.config(text="MENGANALISIS FITUR...", state=tk.DISABLED, bg="#243B53")
            self.root.update_idletasks()

            features = extract_features(self.current_image_path)
            if features is None:
                messagebox.showerror("Error", "Gagal mengekstrak fitur gambar.")
                self.predict_btn.config(text="JALANKAN PREDIKSI AI", state=tk.NORMAL, bg="#00F5A0")
                return

            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])
            confidence = float(np.max(probabilities)) * 100

            pred_text = str(prediction).upper()
            is_fresh = "FRESH" in pred_text or "SEGAR" in pred_text
            
            status_color = "#00F5A0" if is_fresh else "#FF5252"
            
            self.status_value_label.config(text=pred_text, fg=status_color)
            self.conf_value_label.config(text=f"{confidence:.2f}%")
            self.update_progress_bar(confidence)
            
            self.predict_btn.config(text="JALANKAN PREDIKSI AI", state=tk.NORMAL, bg="#00F5A0")
            
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan sistem: {e}")
            self.predict_btn.config(text="JALANKAN PREDIKSI AI", state=tk.NORMAL, bg="#00F5A0")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = UltraModernFruitDetectorApp(root)
    root.mainloop()