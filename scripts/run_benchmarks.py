"""Run benchmarking for selected models and save results to CSV.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --output benchmark_results.csv
    python scripts/run_benchmarks.py --validation-dir path/to/valid

This script benchmarks:
    - Baseline CNN
    - EfficientNetB0
    - MobileNetV3-S
    - MobileNetV3-L

Metrics include:
    - Parameter count
    - Estimated model size
    - Average inference latency
    - Validation accuracy
    - Macro F1
    - Weighted F1
    - Top-3 accuracy
"""

from pathlib import Path
import argparse
from pprint import pprint

from src.training.benchmark import (
    benchmark_model,
    save_benchmark_results,
)

from src.training.evaluate import (
    build_evaluation_generator,
)

from src.training.train import (
    build_baseline_cnn_model,
    build_efficientnetb0_model,
    build_mobilenetv3_model,
)


MODEL_BUILDERS = {
    "CNN": build_baseline_cnn_model,
    "EfficientNetB0": build_efficientnetb0_model,
    "MobileNetV3-S": lambda: build_mobilenetv3_model(
        model_type="small"
    ),
    "MobileNetV3-L": lambda: build_mobilenetv3_model(
        model_type="large"
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark plant disease classification models."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_BUILDERS.keys()),
        default=list(MODEL_BUILDERS.keys()),
        help="Models to benchmark.",
    )

    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help=(
            "Validation directory containing one subdirectory "
            "per class. If omitted, only model performance "
            "characteristics are benchmarked."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Validation batch size.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # VALIDATE INPUTS
    # ---------------------------------------------------------

    if args.validation_dir is not None:
        if not args.validation_dir.is_dir():
            raise FileNotFoundError(
                f"Validation directory not found: "
                f"{args.validation_dir}"
            )

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    print("=" * 70)
    print("PLANT DISEASE MODEL BENCHMARK")
    print("=" * 70)

    print(
        "Models:      "
        + ", ".join(args.models)
    )

    print(
        f"Validation:  "
        f"{args.validation_dir or 'None'}"
    )

    print(
        f"Batch size:  {args.batch_size}"
    )

    print(
        f"Output:      {args.output}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # BENCHMARK
    # ---------------------------------------------------------

    results = []

    for model_name in args.models:

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"Benchmarking: {model_name}"
        )

        print(
            f"{'-' * 70}"
        )

        builder = MODEL_BUILDERS[model_name]

        validation_generator = None

        if args.validation_dir is not None:

            print(
                "Preparing validation generator..."
            )

            # The project's standard evaluation generator already
            # applies the same /255 scaling used by the current
            # training/validation pipeline.
            validation_generator = (
                build_evaluation_generator(
                    data_dir=str(
                        args.validation_dir
                    ),
                    image_size=(224, 224),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
            )

            print(
                "Validation images: "
                f"{validation_generator.samples}"
            )

        result = benchmark_model(
            model_name=model_name,
            model_builder=builder,
            input_shape=(1, 224, 224, 3),
            training_generator=None,
            validation_generator=validation_generator,
        )

        results.append(result)

        pprint(result)

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = save_benchmark_results(
        results,
        args.output,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()