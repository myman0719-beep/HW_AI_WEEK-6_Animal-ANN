import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import json
import os

from tensorflow.keras.models import load_model


# ======================================
# LOAD MODEL
# ======================================

MODEL_PATH = "animal_ann_model.keras"

LABEL_PATH = "animal_labels.json"

IMG_SIZE = 28


# load model
model = load_model(MODEL_PATH)

# load labels
with open(LABEL_PATH, "r") as f:
    data = json.load(f)

index_to_label = {}

for key, value in data.items():
    index_to_label[int(key)] = value


# tên tiếng việt
VIETNAMESE_LABELS = {
    "dog": "chó",
    "cat": "mèo",
    "elephant": "voi",
    "mouse": "chuột",
    "pig": "heo"
}


# ======================================
# APP
# ======================================

class AnimalAIApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Animal Sketch AI")

        self.root.geometry("720x760")

        self.root.resizable(False, False)

        self.root.configure(bg="#EAF6FF")

        self.canvas_size = 420

        # ================= TITLE =================

        self.title_label = tk.Label(
            root,
            text="ANIMAL SKETCH RECOGNIZER",
            font=("Arial", 24, "bold"),
            bg="#EAF6FF",
            fg="#1E3A5F"
        )

        self.title_label.pack(pady=15)

        # ================= NOTE =================

        self.note_label = tk.Label(
            root,
            text="Vẽ: mèo, chó, voi, chuột hoặc heo",
            font=("Arial", 13),
            bg="#EAF6FF",
            fg="#4A6572"
        )

        self.note_label.pack()

        # ================= CANVAS =================

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#111827",
            cursor="pencil",
            highlightthickness=0,
            bd=0
        )

        self.canvas.pack(pady=20)

        self.image = Image.new(
            "L",
            (self.canvas_size, self.canvas_size),
            0
        )

        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.canvas.bind(
            "<B1-Motion>",
            self.paint
        )

        self.canvas.bind(
            "<ButtonRelease-1>",
            self.reset_position
        )

        # ================= RESULT =================

        self.result_label = tk.Label(
            root,
            text="Kết quả dự đoán: ...",
            font=("Arial", 18, "bold"),
            bg="#EAF6FF",
            fg="#0F172A"
        )

        self.result_label.pack(pady=10)

        # ================= BUTTON FRAME =================

        button_frame = tk.Frame(
            root,
            bg="#EAF6FF"
        )

        button_frame.pack(pady=15)

        # ================= PREDICT BUTTON =================

        self.predict_button = tk.Button(
            button_frame,
            text="Nhận diện",
            font=("Arial", 13, "bold"),
            width=14,
            bg="#2563EB",
            fg="white",
            activebackground="#1D4ED8",
            relief="flat",
            command=self.predict
        )

        self.predict_button.grid(
            row=0,
            column=0,
            padx=10
        )

        # ================= CLEAR BUTTON =================

        self.clear_button = tk.Button(
            button_frame,
            text="Xóa",
            font=("Arial", 13, "bold"),
            width=14,
            bg="#EF4444",
            fg="white",
            activebackground="#DC2626",
            relief="flat",
            command=self.clear
        )

        self.clear_button.grid(
            row=0,
            column=1,
            padx=10
        )

    # ======================================
    # VẼ
    # ======================================

    def paint(self, event):

        x = event.x
        y = event.y

        if self.last_x is not None and self.last_y is not None:

            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                fill="#22D3EE",
                width=14,
                capstyle=tk.ROUND,
                smooth=True
            )

            self.draw.line(
                (self.last_x, self.last_y, x, y),
                fill=255,
                width=14
            )

        self.last_x = x
        self.last_y = y

    def reset_position(self, event):

        self.last_x = None
        self.last_y = None

    # ======================================
    # CLEAR
    # ======================================

    def clear(self):

        self.canvas.delete("all")

        self.image = Image.new(
            "L",
            (self.canvas_size, self.canvas_size),
            0
        )

        self.draw = ImageDraw.Draw(self.image)

        self.result_label.config(
            text="Kết quả dự đoán: ..."
        )

    # ======================================
    # PREPROCESS
    # ======================================

    def preprocess_image(self):

        img = self.image

        bbox = img.getbbox()

        if bbox is not None:

            img = img.crop(bbox)

            w, h = img.size

            size = max(w, h) + 80

            new_img = Image.new(
                "L",
                (size, size),
                0
            )

            paste_x = (size - w) // 2
            paste_y = (size - h) // 2

            new_img.paste(
                img,
                (paste_x, paste_y)
            )

            img = new_img

        img = img.resize((IMG_SIZE, IMG_SIZE))

        img = ImageOps.autocontrast(img)

        arr = np.array(img).astype("float32") / 255.0

        arr = arr.reshape(
            1,
            IMG_SIZE * IMG_SIZE
        )

        return arr

    # ======================================
    # PREDICT
    # ======================================

    def predict(self):

        arr = self.preprocess_image()

        preds = model.predict(
            arr,
            verbose=0
        )[0]

        best_index = int(np.argmax(preds))

        best_label = index_to_label[best_index]

        vietnamese_name = VIETNAMESE_LABELS.get(
            best_label,
            best_label
        )

        confidence = preds[best_index] * 100

        self.result_label.config(
            text=f"Dự đoán: {vietnamese_name} ({confidence:.2f}%)"
        )


# ======================================
# RUN APP
# ======================================

if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):

        messagebox.showerror(
            "Lỗi",
            "Không tìm thấy animal_ann_model.keras"
        )

    elif not os.path.exists(LABEL_PATH):

        messagebox.showerror(
            "Lỗi",
            "Không tìm thấy animal_labels.json"
        )

    else:

        root = tk.Tk()

        app = AnimalAIApp(root)

        root.mainloop()
