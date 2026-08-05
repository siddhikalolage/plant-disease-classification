import numpy as np

from src.training.benchmark import benchmark_model
from src.training.train import build_baseline_cnn_model


def test_benchmark_model_runs_without_validation():
    result = benchmark_model(
        model_name='baseline_cnn',
        model_builder=build_baseline_cnn_model,
        input_shape=(1, 224, 224, 3),
        training_generator=None,
        validation_generator=None,
    )

    assert result.model_name == 'baseline_cnn'
    assert result.parameters > 0
    assert result.estimated_size_bytes > 0
    assert result.average_latency_ms > 0.0
    assert result.accuracy is None
    assert result.top_3_accuracy is None
