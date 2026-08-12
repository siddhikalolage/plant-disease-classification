from pathlib import Path
import argparse
import csv

import numpy as np
import tensorflow as tf

from src.training.train import build_training_generators


def class_names_from_generator(generator):
    return [
        name
        for name, _ in sorted(
            generator.class_indices.items(),
            key=lambda item: item[1],
        )
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze misclassified plant disease images."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained .keras model.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing train/ and valid/ directories.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/error_analysis"),
        help="Directory where error analysis results will be saved.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Validation batch size.",
    )

    args = parser.parse_args()

    model_path = args.model
    dataset_root = args.dataset_root
    output_dir = args.output_dir

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {valid_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PLANT DISEASE ERROR ANALYSIS")
    print("=" * 70)

    print(f"Model:            {model_path}")
    print(f"Dataset:          {dataset_root}")
    print(f"Validation:       {valid_dir}")
    print(f"Output directory: {output_dir}")

    print("\nLoading model...")

    model = tf.keras.models.load_model(model_path)

    _, valid_generator = build_training_generators(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        image_size=(224, 224),
        batch_size=args.batch_size,
    )

    valid_generator.shuffle = False
    valid_generator.reset()

    class_names = class_names_from_generator(
        valid_generator
    )

    print(f"Validation images: {valid_generator.samples}")
    print(f"Classes:           {len(class_names)}")

    print("\nGenerating predictions...")

    probabilities = model.predict(
        valid_generator,
        steps=len(valid_generator),
        verbose=1,
    )

    y_true = np.asarray(
        valid_generator.classes
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction count mismatch: "
            f"{len(y_true)} labels vs "
            f"{len(y_pred)} predictions."
        )

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

    output_csv = (
        output_dir / "misclassified_images.csv"
    )

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

    for index, (
        (true_class, predicted_class),
        count,
    ) in enumerate(
        top_pairs[:20],
        start=1,
    ):

        print(
            f"{index:2d}. "
            f"{true_class} -> "
            f"{predicted_class}: "
            f"{count} images"
        )

    # --------------------------------------------------------
    # HIGH-CONFIDENCE WRONG PREDICTIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP HIGH-CONFIDENCE WRONG PREDICTIONS")
    print("=" * 70)

    for index, row in enumerate(
        rows[:20],
        start=1,
    ):

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

    total_images = len(y_true)
    misclassified = len(rows)

    accuracy = (
        1.0 - misclassified / total_images
        if total_images > 0
        else 0.0
    )

    print(
        f"Total validation images: "
        f"{total_images}"
    )

    print(
        f"Misclassified images: "
        f"{misclassified}"
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