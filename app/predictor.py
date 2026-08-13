from pathlib import Path
from typing import List, Union

import numpy as np
import tensorflow as tf
from PIL import Image

from src.constants import CLASS_NAMES, TARGET_IMAGE_SIZE


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = "mobilenet_benchmark_final.keras"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / DEFAULT_MODEL_NAME


ImageInput = Union[str, Path, Image.Image]


def get_model_path(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Path:
    """Resolve the model path relative to the project root."""

    path = Path(model_path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path


def load_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> tf.keras.Model:
    """Load the trained plant-disease classifier."""

    path = get_model_path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}"
        )

    model = tf.keras.models.load_model(
        str(path)
    )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "Model/class mapping mismatch. "
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
        )

    return model


def _load_image(
    image_input: ImageInput,
) -> Image.Image:
    """Load an image from a path or PIL image."""

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    path = Path(image_input)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found at: {path}"
        )

    image = Image.open(path)

    return image.convert("RGB")


def preprocess_image(
    image_input: ImageInput,
    target_size: tuple[int, int] = TARGET_IMAGE_SIZE,
) -> np.ndarray:
    """
    Prepare an image exactly like the validation pipeline.

    The final MobileNetV3 model was evaluated using images
    scaled from [0, 255] to [0, 1] by ImageDataGenerator.

    The model itself contains its MobileNet preprocessing,
    so we only perform the generator-side /255 scaling here.
    """

    image = _load_image(image_input)

    image = image.resize(
        target_size,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array /= 255.0

    return np.expand_dims(
        image_array,
        axis=0,
    )


def predict_probabilities(
    image_input: ImageInput,
    model: tf.keras.Model | None = None,
    model_path: Union[str, Path] | None = None,
) -> np.ndarray:
    """Return the complete 38-class probability vector."""

    if model is None:

        if model_path is not None:
            model = load_model(model_path)

        else:
            model = load_model()

    image_array = preprocess_image(
        image_input
    )

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    return probabilities


def human_readable_label(
    label: str,
) -> str:
    """Convert dataset class names into readable labels."""

    plant, separator, disease = label.partition(
        "___"
    )

    plant_text = (
        plant
        .replace("_", " ")
        .title()
    )

    if not separator:
        return plant_text

    disease_text = (
        disease
        .replace("_", " ")
        .title()
    )

    return f"{plant_text} — {disease_text}"


def get_top_predictions(
    probabilities: np.ndarray,
    top_k: int = 5,
):
    """Return the top-k predictions."""

    top_k = max(
        1,
        min(
            top_k,
            len(probabilities),
        ),
    )

    indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    predictions = []

    for index in indices:

        predictions.append(
            {
                "index": int(index),
                "class_name": CLASS_NAMES[
                    int(index)
                ],
                "label": human_readable_label(
                    CLASS_NAMES[int(index)]
                ),
                "confidence": float(
                    probabilities[int(index)]
                ),
            }
        )

    return predictions


def confidence_status(
    confidence: float,
) -> str:
    """
    Convert confidence into an application-level status.

    Note:
    This is a UI confidence rule, not a calibrated probability.
    """

    if confidence >= 0.90:
        return "high"

    if confidence >= 0.70:
        return "medium"

    return "low"


def predict_with_confidence(
    image_input: ImageInput,
    model: tf.keras.Model | None = None,
    model_path: Union[str, Path] | None = None,
    top_k: int = 5,
):
    """
    Return the primary prediction plus Top-K alternatives.
    """

    probabilities = predict_probabilities(
        image_input=image_input,
        model=model,
        model_path=model_path,
    )

    predictions = get_top_predictions(
        probabilities,
        top_k=top_k,
    )

    best = predictions[0]

    return (
        best["index"],
        best["confidence"],
        predictions,
        confidence_status(
            best["confidence"]
        ),
    )


def predict(
    image_input: ImageInput,
    model: tf.keras.Model | None = None,
    model_path: Union[str, Path] | None = None,
) -> int:
    """Return the predicted class index."""

    prediction_index, _, _, _ = (
        predict_with_confidence(
            image_input=image_input,
            model=model,
            model_path=model_path,
            top_k=1,
        )
    )

    return prediction_index


def get_class_name(
    index: int,
) -> str:
    """Return the raw class name for a class index."""

    if 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]

    return "Unknown"


def predict_from_path(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
):
    """
    Convenience function for command-line or application use.
    """

    return predict_with_confidence(
        image_input=image_path,
        model_path=model_path,
        top_k=5,
    )