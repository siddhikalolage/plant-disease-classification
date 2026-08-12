from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.training.train import build_training_generators


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32


# ============================================================
# HELPERS
# ============================================================

def load_validation_generator(dataset_root: Path, batch_size: int):
    """Build the validation generator using the project's pipeline."""

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {valid_dir}"
        )

    _, valid_generator = build_training_generators(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=batch_size,
    )

    # Evaluation must remain deterministic.
    valid_generator.shuffle = False
    valid_generator.reset()

    return valid_generator


def get_class_names(generator):
    """Recover class names in their numeric class-index order."""

    ordered = sorted(
        generator.class_indices.items(),
        key=lambda item: item[1],
    )

    return [name for name, _ in ordered]


def predict_full_validation_set(model, generator):
    """Generate predictions for every validation image."""

    generator.reset()

    probabilities = model.predict(
        generator,
        steps=len(generator),
        verbose=1,
    )

    probabilities = np.asarray(probabilities)

    y_true = np.asarray(
        generator.classes,
        dtype=np.int64,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            "Prediction/label count mismatch: "
            f"{len(y_true)} labels vs "
            f"{len(y_pred)} predictions."
        )

    return y_true, y_pred, probabilities


def calculate_metrics(
    y_true,
    y_pred,
    class_names,
):
    """Calculate overall and per-class classification metrics."""

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
    }

    return metrics


def calculate_confusion_matrix(
    y_true,
    y_pred,
    class_names,
):
    """Create the complete 38 x 38 confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    return matrix


def get_worst_classes(
    report,
    class_names,
    limit=10,
):
    """Return the classes with the weakest F1 scores."""

    results = []

    for class_name in class_names:
        values = report.get(class_name)

        if not values:
            continue

        results.append(
            {
                "class": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )

    results.sort(
        key=lambda item: item["f1"]
    )

    return results[:limit]


def get_top_confusions(
    confusion,
    class_names,
    limit=15,
):
    """Find the strongest off-diagonal confusion pairs."""

    pairs = []

    for true_index in range(len(class_names)):

        for predicted_index in range(len(class_names)):

            if true_index == predicted_index:
                continue

            count = int(
                confusion[
                    true_index,
                    predicted_index,
                ]
            )

            if count == 0:
                continue

            pairs.append(
                {
                    "true_class": class_names[true_index],
                    "predicted_class": class_names[predicted_index],
                    "count": count,
                }
            )

    pairs.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return pairs[:limit]


def save_json(data, output_path: Path):
    """Save structured evaluation results as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    model_path = Path(
        "models/mobilenet_benchmark_final.keras"
    )

    dataset_root = Path(
        r"C:\Users\siddh\Downloads\plant dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
    )

    output_dir = Path("reports")

    print("=" * 70)
    print("FULL PLANT DISEASE MODEL EVALUATION")
    print("=" * 70)

    print(f"Model:       {model_path}")
    print(f"Dataset:     {dataset_root}")
    print(f"Output:      {output_dir}")

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        f"Model output: {model.output_shape}"
    )

    if model.output_shape[-1] != 38:
        raise ValueError(
            "Expected a 38-class classifier, "
            f"but received {model.output_shape}."
        )

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    print("\nLoading validation generator...")

    generator = load_validation_generator(
        dataset_root=dataset_root,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    class_names = get_class_names(
        generator
    )

    print(
        f"Validation images: {generator.samples}"
    )

    print(
        f"Classes: {len(class_names)}"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_true, y_pred, probabilities = (
        predict_full_validation_set(
            model=model,
            generator=generator,
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print("\nCalculating metrics...")

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    confusion = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    worst_classes = get_worst_classes(
        metrics["classification_report"],
        class_names,
        limit=10,
    )

    top_confusions = get_top_confusions(
        confusion,
        class_names,
        limit=15,
    )

    # --------------------------------------------------------
    # CONSOLE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:          {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision:   {metrics['precision_macro']:.4f}"
    )

    print(
        f"Macro Recall:      {metrics['recall_macro']:.4f}"
    )

    print(
        f"Macro F1:          {metrics['f1_macro']:.4f}"
    )

    print(
        f"Weighted Precision:{metrics['precision_weighted']:.4f}"
    )

    print(
        f"Weighted Recall:   {metrics['recall_weighted']:.4f}"
    )

    print(
        f"Weighted F1:       {metrics['f1_weighted']:.4f}"
    )

    # --------------------------------------------------------
    # WORST CLASSES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("10 WEAKEST CLASSES")
    print("=" * 70)

    for index, item in enumerate(
        worst_classes,
        start=1,
    ):

        print(
            f"{index:2d}. "
            f"{item['class']} | "
            f"Precision={item['precision']:.4f} | "
            f"Recall={item['recall']:.4f} | "
            f"F1={item['f1']:.4f} | "
            f"Support={item['support']}"
        )

    # --------------------------------------------------------
    # TOP CONFUSIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP CONFUSION PAIRS")
    print("=" * 70)

    for index, item in enumerate(
        top_confusions,
        start=1,
    ):

        print(
            f"{index:2d}. "
            f"{item['true_class']} "
            f"--> "
            f"{item['predicted_class']} "
            f"({item['count']} images)"
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    complete_report = {
        "model": str(model_path),
        "dataset": str(dataset_root),
        "validation_samples": int(generator.samples),
        "num_classes": len(class_names),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "class_names": class_names,
        "classification_report": metrics[
            "classification_report"
        ],
        "worst_classes": worst_classes,
        "top_confusions": top_confusions,
        "confusion_matrix": confusion.tolist(),
    }

    report_path = (
        output_dir /
        "full_evaluation_report.json"
    )

    save_json(
        complete_report,
        report_path,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved report: {report_path}"
    )


if __name__ == "__main__":
    main()