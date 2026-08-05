from pathlib import Path

from app.predictor import predict, get_class_name


def predict_from_path(image_path: str, model_path: str | Path = "models/plant_disease_cnn_model.keras") -> str:
    prediction_idx = predict(image_path, model_path=model_path)
    return get_class_name(prediction_idx)
