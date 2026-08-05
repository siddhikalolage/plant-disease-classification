import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf

DEFAULT_MODEL_NAME = "plant_disease_cnn_model.keras"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / DEFAULT_MODEL_NAME
CLASS_NAMES: List[str] = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def get_model_path(model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image.reshape(1, target_size[0], target_size[1], 3)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    path = get_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(str(path))


def predict(image_path: str, model: tf.keras.Model | None = None) -> int:
    if model is None:
        model = load_model()
    features = preprocess_image(image_path)
    prediction = np.argmax(model.predict(features), axis=-1)[0]
    return int(prediction)


def get_class_name(index: int) -> str:
    return CLASS_NAMES[index] if 0 <= index < len(CLASS_NAMES) else "Unknown"
