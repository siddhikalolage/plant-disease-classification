# Plant Disease Classification using CNN

## Overview
This project uses Deep Learning to detect plant leaf diseases from images. A Convolutional Neural Network (CNN) is trained to classify leaf images into healthy and diseased categories, helping in early disease identification.

## Problem Statement
Plant diseases reduce crop productivity and require early detection. Manual inspection is inefficient and error-prone. This project automates disease detection using image classification.

## Approach
- Built a CNN model for image classification
- Performed image preprocessing and normalization
- Applied data augmentation to improve model generalization
- Evaluated performance using accuracy and confusion matrix

## Dataset
Dataset sourced from Kaggle:  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib

## Results
- Achieved approximately 90% classification accuracy
- Improved robustness using data augmentation techniques

## Benchmarking and Evaluation
- Added benchmark utilities in `src/training/benchmark.py`
- Supports parameter count, estimated model size, inference latency, and optional validation metrics
- Run from the repo root with `PYTHONPATH=. python -c "from src.training.benchmark import build_benchmark_suite; ..."`
- Example usage:

```bash
set PYTHONPATH=.
python -c "from src.training.benchmark import build_benchmark_suite; from src.inference.predict import build_baseline_cnn_model; results = build_benchmark_suite({'CNN': build_baseline_cnn_model}); print(results)"
```

Notebook / analysis script:

- `notebooks/benchmark_analysis.py`: small analysis helper that loads `benchmark_results.csv`, prints a summary, and saves comparison plots to `notebooks/figures/`.

Example run to generate results and analyze:

```bash
set PYTHONPATH=.
python scripts/run_benchmarks.py --validation-dir path\to\val --output benchmark_results.csv
python notebooks/benchmark_analysis.py --input benchmark_results.csv --show
```
## Key Learnings
- CNN architecture and training
- Image preprocessing and augmentation
- Model evaluation techniques
