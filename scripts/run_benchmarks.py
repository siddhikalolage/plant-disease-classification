"""Run benchmarking for selected models and save results to CSV.

Usage:
    python scripts/run_benchmarks.py --output benchmark_results.csv --validation-dir path/to/val

This script builds baseline CNN, EfficientNetB0, MobileNetV3-S, and MobileNetV3-L
and measures parameter count, estimated size, avg inference latency, and (optionally)
validation metrics if a validation dataset is provided.
"""
from pathlib import Path
import argparse
import json
from pprint import pprint

from src.training.benchmark import benchmark_model, save_benchmark_results
from src.training.evaluate import build_evaluation_generator
from src.inference.predict import (
    build_baseline_cnn_model,
    build_efficientnetb0_model,
    build_mobilenetv3_model,
)
from app.predictor import get_model_preprocessing


MODEL_BUILDERS = {
    'CNN': build_baseline_cnn_model,
    'EfficientNetB0': build_efficientnetb0_model,
    'MobileNetV3-S': lambda: build_mobilenetv3_model(model_type='small'),
    'MobileNetV3-L': lambda: build_mobilenetv3_model(model_type='large'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=list(MODEL_BUILDERS.keys()), default=list(MODEL_BUILDERS.keys()))
    parser.add_argument('--validation-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=str, default='benchmark_results.csv')
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        builder = MODEL_BUILDERS[model_name]

        # Create a small model instance to infer preprocessing function for the generator
        temp_model = builder()
        preprocessing_fn = get_model_preprocessing(temp_model)

        validation_generator = None
        if args.validation_dir:
            validation_generator = build_evaluation_generator(
                args.validation_dir,
                image_size=(224, 224),
                batch_size=args.batch_size,
                shuffle=False,
                preprocessing_function=preprocessing_fn,
            )

        print(f'Benchmarking {model_name}...')
        result = benchmark_model(
            model_name=model_name,
            model_builder=builder,
            input_shape=(1, 224, 224, 3),
            training_generator=None,
            validation_generator=validation_generator,
        )
        results.append(result)
        pprint(result)

    out_path = save_benchmark_results(results, Path(args.output))
    print(f'Benchmark results saved to: {out_path}')


if __name__ == '__main__':
    main()
