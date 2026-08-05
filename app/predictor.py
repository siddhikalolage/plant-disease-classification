from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

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
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def get_model_path(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def preprocess_image(image_input: Union[str, Path, Image.Image], target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {image_input}")
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input.convert("RGB"))
    else:
        raise TypeError("Unsupported image input type. Provide a file path or PIL.Image.Image.")

    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image.reshape(1, target_size[0], target_size[1], 3)


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    path = get_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(str(path))


def human_readable_label(label: str) -> str:
    plant, _, disease = label.partition('___')
    plant_text = plant.replace('_', ' ').title()
    disease_text = disease.replace('_', ' ').title()
    return f"{plant_text} — {disease_text}" if disease else plant_text


def predict(image_input: Union[str, Path, Image.Image], model: tf.keras.Model | None = None) -> int:
    if model is None:
        model = load_model()
    features = preprocess_image(image_input)
    prediction = np.argmax(model.predict(features), axis=-1)[0]
    return int(prediction)


def predict_with_confidence(image_input: Union[str, Path, Image.Image], model: tf.keras.Model | None = None, top_k: int = 3):
    if model is None:
        model = load_model()
    features = preprocess_image(image_input)
    probabilities = model.predict(features)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        (human_readable_label(CLASS_NAMES[int(idx)]), float(probabilities[int(idx)]))
        for idx in top_indices
    ]
    return int(top_indices[0]), float(probabilities[int(top_indices[0])]), top_predictions


def find_last_conv_layer(model: tf.keras.Model) -> str:
    def scan_layers(layers_list):
        for layer in reversed(layers_list):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return layer.name
            if isinstance(layer, tf.keras.Model):
                nested = scan_layers(layer.layers)
                if nested:
                    return nested
        return None

    last_conv_name = scan_layers(model.layers)
    if not last_conv_name:
        raise ValueError('Unable to find a convolutional layer in the model for Grad-CAM.')
    return last_conv_name


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, pred_index: int = None, last_conv_layer_name: str | None = None) -> np.ndarray:
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()


def overlay_gradcam(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(original_image.convert('RGB'))
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(heatmap, alpha, original, 1 - alpha, 0)
    return Image.fromarray(overlay)


def predict_with_gradcam(image_input: Union[str, Path, Image.Image], model: tf.keras.Model | None = None, top_k: int = 3):
    if model is None:
        model = load_model()

    if isinstance(image_input, (str, Path)):
        original_image = Image.open(image_input).convert('RGB')
    else:
        original_image = image_input.convert('RGB')

    features = preprocess_image(original_image)
    probabilities = model.predict(features)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        (human_readable_label(CLASS_NAMES[int(idx)]), float(probabilities[int(idx)]))
        for idx in top_indices
    ]

    heatmap = make_gradcam_heatmap(features, model, pred_index=int(top_indices[0]))
    overlay = overlay_gradcam(original_image, heatmap)
    return int(top_indices[0]), float(probabilities[int(top_indices[0])]), top_predictions, overlay


def get_class_name(index: int) -> str:
    return CLASS_NAMES[index] if 0 <= index < len(CLASS_NAMES) else "Unknown"
