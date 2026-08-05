import cv2
import numpy as np


def preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image.reshape(1, target_size[0], target_size[1], 3)
