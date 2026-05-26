import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
to_categorical = tf.keras.utils.to_categorical
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
to_categorical = tf.keras.utils.to_categorical

from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIG
# ==========================================================

IMG_SIZE = 28

CLASSES = [
    "heo",
    "meo",
    "chuot",
    "cho",
    "voi"
]

DATASET_PATH = "dataset"

# ==========================================================
# LOAD DATA
# ==========================================================

X = []
y = []

for label, class_name in enumerate(CLASSES):

    folder_path = os.path.join(DATASET_PATH, class_name)

    for file_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize 28x28
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # invert color
        gray = 255 - gray

        # normalize
        gray = gray.astype("float32") / 255.0

        # flatten
        gray = gray.reshape(IMG_SIZE * IMG_SIZE)

        X.append(gray)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

# ==========================================================
# ONE HOT ENCODING
# ==========================================================

y = to_categorical(y, num_classes=len(CLASSES))

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# BUILD ANN MODEL
# ==========================================================

model = Sequential()

model.add(Dense(
    512,
    activation="relu",
    input_shape=(IMG_SIZE * IMG_SIZE,)
))

model.add(Dense(
    256,
    activation="relu"
))

model.add(Dense(
    len(CLASSES),
    activation="softmax"
))

# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================================
# TRAIN
# ==========================================================

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================================================
# EVALUATE
# ==========================================================

loss, acc = model.evaluate(X_test, y_test)

print("Accuracy:", acc)

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save("animal_ann.h5")

print("Model saved: animal_ann.h5")