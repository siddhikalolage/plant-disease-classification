import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import tensorflow as tf

from src.training.evaluate import compute_classification_metrics, compute_roc_auc, compute_top_k_accuracy, get_labels


@dataclass
class BenchmarkResult:
    model_name: str
    parameters: int
    estimated_size_bytes: int
    average_latency_ms: float
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    top_3_accuracy: Optional[float] = None


def estimate_model_size(model: tf.keras.Model) -> int:
    total_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights + model.non_trainable_weights])
    )
    return int(total_params * 4)


def measure_inference_latency(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int, int] = (1, 224, 224, 3),
    warmup_runs: int = 5,
    measured_runs: int = 20,
) -> float:
    sample_input = np.random.rand(*input_shape).astype("float32")
    for _ in range(warmup_runs):
        model.predict(sample_input, verbose=0)

    timings = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        model.predict(sample_input, verbose=0)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings))


def benchmark_model(
    model_name: str,
    model_builder: Callable[[], tf.keras.Model],
    input_shape: tuple[int, int, int, int] = (1, 224, 224, 3),
    training_generator=None,
    validation_generator=None,
) -> BenchmarkResult:
    model = model_builder()
    parameters = int(model.count_params())
    size_bytes = estimate_model_size(model)
    latency_ms = measure_inference_latency(model, input_shape=input_shape)

    result = BenchmarkResult(
        model_name=model_name,
        parameters=parameters,
        estimated_size_bytes=size_bytes,
        average_latency_ms=latency_ms,
    )

    if validation_generator is not None:
        probabilities = model.predict(validation_generator, verbose=0)
        y_pred = np.argmax(probabilities, axis=-1)
        y_true = validation_generator.classes
        labels = get_labels(validation_generator)

        metrics = compute_classification_metrics(y_true, y_pred)
        result.accuracy = metrics['accuracy']
        result.macro_f1 = metrics['f1_macro']
        result.weighted_f1 = metrics['f1_weighted']
        result.top_3_accuracy = compute_top_k_accuracy(y_true, probabilities, k=3)

    return result


def build_benchmark_suite(model_builders: Dict[str, Callable[[], tf.keras.Model]], **kwargs):
    results = []
    for model_name, builder in model_builders.items():
        result = benchmark_model(model_name, builder, **kwargs)
        results.append(result)
    return results


def save_benchmark_results(results: list[BenchmarkResult], output_file: Optional[Path] = None):
    if output_file is None:
        output_file = Path.cwd() / "benchmark_results.csv"
    lines = [
        "model_name,parameters,estimated_size_bytes,average_latency_ms,accuracy,macro_f1,weighted_f1,top_3_accuracy"
    ]
    for result in results:
        lines.append(
            f"{result.model_name},{result.parameters},{result.estimated_size_bytes},{result.average_latency_ms:.3f},"
            f"{result.accuracy if result.accuracy is not None else ''},"
            f"{result.macro_f1 if result.macro_f1 is not None else ''},"
            f"{result.weighted_f1 if result.weighted_f1 is not None else ''},"
            f"{result.top_3_accuracy if result.top_3_accuracy is not None else ''}"
        )
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file
