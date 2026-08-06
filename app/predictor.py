import sys
from pathlib import Path
from typing import Callable, List, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import efficientnet, mobilenet_v2, mobilenet_v3

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import CLASS_NAMES, TARGET_IMAGE_SIZE

DEFAULT_MODEL_NAME = "plant_disease_cnn_model.keras"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / DEFAULT_MODEL_NAME


def get_model_path(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def get_model_preprocessing(model: tf.keras.Model) -> Callable[[np.ndarray], np.ndarray]:
    def scan_model_layers(layer_list):
        for layer in layer_list:
            layer_name = getattr(layer, 'name', '').lower()
            if 'efficientnet' in layer_name:
                return efficientnet.preprocess_input
            if 'mobilenetv3' in layer_name:
                return mobilenet_v3.preprocess_input
            if 'mobilenetv2' in layer_name:
                return mobilenet_v2.preprocess_input
            if isinstance(layer, tf.keras.Model):
                nested = scan_model_layers(layer.layers)
                if nested:
                    return nested
        return None

    preprocess_fn = scan_model_layers(model.layers)
    return preprocess_fn if preprocess_fn is not None else lambda x: x / 255.0


def preprocess_image(image_input: Union[str, Path, Image.Image], target_size: tuple[int, int] = TARGET_IMAGE_SIZE, model: tf.keras.Model | None = None) -> np.ndarray:
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
    image = image.astype("float32")
    if model is not None:
        preprocessing_fn = get_model_preprocessing(model)
        image = preprocessing_fn(image)
    else:
        image /= 255.0
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


def predict(
    image_input: Union[str, Path, Image.Image],
    model: tf.keras.Model | None = None,
    model_path: Union[str, Path] | None = None,
) -> int:
    if model is None:
        model = load_model(model_path) if model_path is not None else load_model()
    features = preprocess_image(image_input, model=model)
    prediction = np.argmax(model.predict(features), axis=-1)[0]
    return int(prediction)


def predict_with_confidence(image_input: Union[str, Path, Image.Image], model: tf.keras.Model | None = None, top_k: int = 3):
    if model is None:
        model = load_model()
    features = preprocess_image(image_input, model=model)
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


def get_layer_by_name_recursive(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer | None:
    try:
        return model.get_layer(layer_name)
    except (ValueError, AttributeError):
        pass

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            nested = get_layer_by_name_recursive(layer, layer_name)
            if nested is not None:
                return nested
    return None


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, pred_index: int = None, last_conv_layer_name: str | None = None) -> np.ndarray:
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    target_layer = get_layer_by_name_recursive(model, last_conv_layer_name)
    if target_layer is None:
        raise ValueError(f"Unable to resolve layer '{last_conv_layer_name}' in the model for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [target_layer.output, model.output]
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

    features = preprocess_image(original_image, model=model)
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
