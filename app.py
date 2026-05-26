from flask import Flask, render_template, request, jsonify

import numpy as np
import cv2
import base64

import tensorflow as tf

load_model = tf.keras.models.load_model

import tensorflow as tf

load_model = tf.keras.models.load_model
# ==========================================================
# INIT
# ==========================================================

app = Flask(__name__)

model = load_model("animal_ann.h5")

CLASSES = [
    "Heo",
    "Mèo",
    "Chuột",
    "Chó",
    "Voi"
]

IMG_SIZE = 28

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess_image(base64_image):

    image_data = base64_image.split(",")[1]

    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    gray = 255 - gray

    gray = gray.astype("float32") / 255.0

    gray = gray.reshape(1, IMG_SIZE * IMG_SIZE)

    return gray

# ==========================================================
# ROUTE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================================
# PREDICT
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    image = data["image"]

    x = preprocess_image(image)

    prediction = model.predict(x)[0]

    class_index = np.argmax(prediction)

    result = CLASSES[class_index]

    confidence = float(prediction[class_index]) * 100

    return jsonify({
        "animal": result,
        "confidence": round(confidence, 2)
    })

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)