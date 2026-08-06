import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predictor import get_class_name, predict


def predict_from_path(image_path: str, model_path: str | Path = "models/plant_disease_cnn_model.keras") -> str:
    model_path_obj = Path(model_path)
    if not model_path_obj.is_absolute():
        model_path_obj = ROOT_DIR / model_path_obj
    prediction_idx = predict(image_path, model_path=model_path_obj)
    return get_class_name(prediction_idx)
