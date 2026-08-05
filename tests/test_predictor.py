from pathlib import Path

import numpy as np
from PIL import Image

from app.predictor import get_class_name, preprocess_image


def test_get_class_name_valid_index():
    assert get_class_name(0) == 'Apple___Apple_scab'


def test_get_class_name_invalid_index():
    assert get_class_name(999) == 'Unknown'


def test_preprocess_image_with_pil_image_returns_correct_shape():
    image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
    result = preprocess_image(image)
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
