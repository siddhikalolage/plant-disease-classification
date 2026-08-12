from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.training.train import build_training_generators
from src.training.evaluate import (
    evaluate_generator,
    compute_classification_metrics,
    compute_top_k_accuracy,
    compute_roc_auc,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained plant disease classification model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained .keras model.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing train/ and valid/ directories.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Validation batch size.",
    )

    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    valid_dir = args.dataset_root / "valid"

    if not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {valid_dir}"
        )

    print("=" * 60)
    print("PLANT DISEASE MODEL EVALUATION")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Validation:  {valid_dir}")
    print(f"Classes:     {NUM_CLASSES}")
    print("=" * 60)

    model = tf.keras.models.load_model(args.model)

    _, valid_generator = build_training_generators(
        train_dir=str(args.dataset_root / "train"),
        valid_dir=str(valid_dir),
        image_size=(224, 224),
        batch_size=args.batch_size,
    )

    # Make sure predictions cover the entire validation set.
    valid_generator.reset()

    print("\nRunning predictions...")

    probabilities = model.predict(
        valid_generator,
        verbose=1,
    )

    y_true = valid_generator.classes
    y_pred = np.argmax(probabilities, axis=1)

    # Safety check.
    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction count mismatch: "
            f"{len(y_true)} labels vs {len(y_pred)} predictions."
        )

    print("\nComputing metrics...")

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    top3 = compute_top_k_accuracy(
        y_true,
        probabilities,
        k=3,
    )

    top5 = compute_top_k_accuracy(
        y_true,
        probabilities,
        k=5,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"top_3_accuracy: {top3:.4f}")
    print(f"top_5_accuracy: {top5:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()