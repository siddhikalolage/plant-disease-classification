from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

from src.training.train import build_training_generators


MODEL_PATH = Path("models/mobilenet_benchmark_final.keras")

DATASET_ROOT = Path(
    r"C:\Users\siddh\Downloads\plant dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
)

OUTPUT_DIR = Path("reports/error_analysis")


def class_names_from_generator(generator):
    return [
        name
        for name, _ in sorted(
            generator.class_indices.items(),
            key=lambda item: item[1],
        )
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PLANT DISEASE ERROR ANALYSIS")
    print("=" * 70)

    model = tf.keras.models.load_model(MODEL_PATH)

    _, valid_generator = build_training_generators(
        train_dir=str(DATASET_ROOT / "train"),
        valid_dir=str(DATASET_ROOT / "valid"),
        image_size=(224, 224),
        batch_size=32,
    )

    valid_generator.shuffle = False
    valid_generator.reset()

    class_names = class_names_from_generator(valid_generator)

    print(f"Model: {MODEL_PATH}")
    print(f"Validation images: {valid_generator.samples}")
    print(f"Classes: {len(class_names)}")
    print("\nGenerating predictions...")

    probabilities = model.predict(
        valid_generator,
        steps=len(valid_generator),
        verbose=1,
    )

    y_true = valid_generator.classes
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    image_paths = [
        str(path)
        for path in valid_generator.filepaths
    ]

    rows = []

    for index in range(len(y_true)):
        if y_true[index] == y_pred[index]:
            continue

        rows.append(
            {
                "image": image_paths[index],
                "true_class": class_names[y_true[index]],
                "predicted_class": class_names[y_pred[index]],
                "confidence": float(confidences[index]),
            }
        )

    rows.sort(
        key=lambda row: row["confidence"],
        reverse=True,
    )

    output_csv = OUTPUT_DIR / "misclassified_images.csv"

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "true_class",
                "predicted_class",
                "confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # MAJOR CONFUSION PAIRS
    # --------------------------------------------------------

    pair_counts = {}

    for row in rows:
        key = (
            row["true_class"],
            row["predicted_class"],
        )

        pair_counts[key] = (
            pair_counts.get(key, 0) + 1
        )

    top_pairs = sorted(
        pair_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("TOP CONFUSION PAIRS")
    print("=" * 70)

    for index, ((true_class, predicted_class), count) in enumerate(
        top_pairs[:20],
        start=1,
    ):
        print(
            f"{index:2d}. "
            f"{true_class} -> {predicted_class}: "
            f"{count} images"
        )

    # --------------------------------------------------------
    # HIGH-CONFIDENCE WRONG PREDICTIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP HIGH-CONFIDENCE WRONG PREDICTIONS")
    print("=" * 70)

    for index, row in enumerate(rows[:20], start=1):
        print(
            f"{index:2d}. "
            f"{row['true_class']} -> "
            f"{row['predicted_class']} | "
            f"confidence={row['confidence']:.4f}"
        )
        print(
            f"    {row['image']}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Total validation images: "
        f"{len(y_true)}"
    )

    print(
        f"Misclassified images: "
        f"{len(rows)}"
    )

    accuracy = 1.0 - (
        len(rows) / len(y_true)
    )

    print(
        f"Accuracy from predictions: "
        f"{accuracy:.4f}"
    )

    print(
        f"\nSaved: {output_csv}"
    )


if __name__ == "__main__":
    main()