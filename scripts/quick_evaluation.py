from pathlib import Path

import numpy as np
import tensorflow as tf

from src.training.train import build_training_generators


MODEL_PATH = Path("models/mobilenet_benchmark_final.keras")

DATASET_ROOT = Path(
    r"C:\Users\siddh\Downloads\plant dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
)


def main():
    print("=" * 60)
    print("QUICK FINAL MODEL EVALUATION")
    print("=" * 60)

    model = tf.keras.models.load_model(MODEL_PATH)

    _, valid_generator = build_training_generators(
        train_dir=str(DATASET_ROOT / "train"),
        valid_dir=str(DATASET_ROOT / "valid"),
        image_size=(224, 224),
        batch_size=32,
    )

    valid_generator.shuffle = False
    valid_generator.reset()

    print(f"Model: {MODEL_PATH}")
    print(f"Validation images: {valid_generator.samples}")
    print("\nGenerating predictions...")

    probabilities = model.predict(
        valid_generator,
        steps=len(valid_generator),
        verbose=1,
    )

    y_true = valid_generator.classes

    top_1 = np.argmax(probabilities, axis=1)

    top_3 = np.argsort(
        probabilities,
        axis=1,
    )[:, -3:]

    top_5 = np.argsort(
        probabilities,
        axis=1,
    )[:, -5:]

    top_1_accuracy = np.mean(
        top_1 == y_true
    )

    top_3_accuracy = np.mean(
        [
            y_true[i] in top_3[i]
            for i in range(len(y_true))
        ]
    )

    top_5_accuracy = np.mean(
        [
            y_true[i] in top_5[i]
            for i in range(len(y_true))
        ]
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    incorrect = top_1 != y_true

    high_confidence_wrong = np.sum(
        incorrect & (confidence >= 0.90)
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Top-1 accuracy:            {top_1_accuracy:.4f}")
    print(f"Top-3 accuracy:            {top_3_accuracy:.4f}")
    print(f"Top-5 accuracy:            {top_5_accuracy:.4f}")

    print(
        f"\nAverage confidence:        "
        f"{confidence.mean():.4f}"
    )

    print(
        f"Wrong predictions >= 90%: "
        f"{high_confidence_wrong}"
    )

    print(
        f"Total wrong predictions:   "
        f"{incorrect.sum()}"
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()