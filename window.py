import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import threading

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

# ================= CONFIG =================
ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_swin_large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_W, DISPLAY_H = 800, 600

WINDOW_KEYWORDS = ["window", "windowpane", "glass"]
FALLBACK_KEYWORDS = ["curtain", "drapery", "blind", "shutter"]

# ================= APP CLASS =================
class WindowSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.image_np = None
        self.image_pil = None
        self.window_mask = None
        self.processor = None
        self.model = None
        self.model_loaded = False
        self.detection_status = "Waiting..."

        self.setup_ui()
        threading.Thread(target=self.load_model_thread, daemon=True).start()

    # ================= UI =================
    def setup_ui(self):
        tk.Label(
            self.root,
            text="Smart Window & Curtain Detector",
            font=("Helvetica", 16, "bold"),
            bg="#f0f0f0"
        ).pack(pady=10)

        self.status_var = tk.StringVar(value="Loading model...")
        self.lbl_status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Consolas", 10),
            fg="red",
            bg="#f0f0f0"
        )
        self.lbl_status.pack(pady=5)

        self.btn_load = tk.Button(
            self.root,
            text="Load Image",
            command=self.load_image,
            font=("Helvetica", 12),
            width=20,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_load.pack(pady=5)

        self.canvas = tk.Canvas(
            self.root,
            width=DISPLAY_W,
            height=DISPLAY_H,
            bg="#333"
        )
        self.canvas.pack(pady=10)

        self.btn_run = tk.Button(
            self.root,
            text="Run Detection",
            command=self.start_segmentation_thread,
            font=("Helvetica", 12),
            width=30,
            bg="#2196F3",
            fg="white",
            state="disabled"
        )
        self.btn_run.pack(pady=15)

    # ================= MODEL =================
    def load_model_thread(self):
        try:
            self.processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                ONEFORMER_MODEL
            ).to(DEVICE).eval()

            self.model_loaded = True
            self.root.after(0, lambda: self.status_var.set(f"Model loaded on {DEVICE.upper()}"))
            self.root.after(0, lambda: self.lbl_status.config(fg="green"))

            if self.image_pil:
                self.btn_run.config(state="normal")

        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    # ================= UTIL =================
    def find_class_ids(self, keywords):
        found = []
        for cid, label in self.model.config.id2label.items():
            for kw in keywords:
                if kw in label.lower():
                    found.append(cid)
                    break
        return found

    # ================= IMAGE LOAD =================
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        self.image_pil = Image.open(path).convert("RGB")
        self.image_np = np.array(self.image_pil)

        display_img = self.image_pil.resize(
            (DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS
        )
        self.tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.status_var.set("Image loaded")

        if self.model_loaded:
            self.btn_run.config(state="normal")

    # ================= SEGMENTATION =================
    def start_segmentation_thread(self):
        self.btn_run.config(state="disabled", text="Processing...")
        threading.Thread(target=self.run_segmentation, daemon=True).start()

    def run_segmentation(self):
        try:
            w, h = self.image_pil.size

            inputs = self.processor(
                images=self.image_pil,
                task_inputs=["semantic"],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**inputs)

            seg_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(h, w)]
            )[0].cpu().numpy()

            # -------- MASK LOGIC --------
            window_ids = self.find_class_ids(WINDOW_KEYWORDS)
            curtain_ids = self.find_class_ids(FALLBACK_KEYWORDS)

            window_mask = np.isin(seg_map, window_ids).astype(np.uint8)
            curtain_mask = np.isin(seg_map, curtain_ids).astype(np.uint8)

            w_pixels = np.sum(window_mask)
            c_pixels = np.sum(curtain_mask)

            if w_pixels > 0 and c_pixels > 0:
                raw_mask = cv2.bitwise_or(window_mask, curtain_mask)
                self.detection_status = "Window + Curtain (Merged)"

            elif w_pixels > 0:
                raw_mask = window_mask
                self.detection_status = "Window"

            elif c_pixels > 0:
                raw_mask = curtain_mask
                self.detection_status = "Curtain / Blind"

            else:
                raw_mask = None
                self.detection_status = "No target found"

            # -------- POST PROCESS --------
            if raw_mask is not None:
                kernel = np.ones((15, 15), np.uint8)
                self.window_mask = cv2.dilate(raw_mask, kernel, iterations=1)
            else:
                self.window_mask = None

            self.root.after(0, self.show_results)

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            self.root.after(0, lambda: self.btn_run.config(state="normal", text="Run Detection"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

    # ================= VISUAL =================
    def show_results(self):
        final_vis = self.image_np.copy()

        if self.window_mask is not None:
            mask = self.window_mask.astype(bool)
            overlay = np.zeros_like(final_vis)
            overlay[mask] = [0, 255, 0]
            final_vis = cv2.addWeighted(final_vis, 0.7, overlay, 0.3, 0)

        plt.figure(figsize=(10, 6))
        plt.title(self.detection_status, fontsize=12, color="blue")
        plt.imshow(final_vis)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# ================= MAIN =================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Smart Window / Curtain Agent")
    root.geometry("900x750")
    root.configure(bg="#f0f0f0")

    app = WindowSegmentationApp(root)
    root.mainloop()