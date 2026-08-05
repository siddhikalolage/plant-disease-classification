import os
from pathlib import Path

import numpy as np

from src.data.preprocessing import preprocess_image


def test_preprocess_image_creates_array(tmp_path):
    image_path = tmp_path / "sample.png"
    # Create a small sample image using numpy
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
    import cv2
    cv2.imwrite(str(image_path), arr)

    result = preprocess_image(str(image_path))
    assert result.shape == (1, 224, 224, 3)
    assert np.all(result <= 1.0)
    assert np.all(result >= 0.0)
