import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.predictor import get_class_name, predict_with_confidence
from src.training.train import build_baseline_cnn_model, build_training_generators, compile_model, train_model


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def find_sample_image(root_dir: Path) -> Path:
    for path in root_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
            return path
    raise FileNotFoundError(f'No image files found under {root_dir}')


def build_subset_dataset(dataset_root: Path, output_root: Path, class_names: list[str], train_per_class: int = 20, valid_per_class: int = 5) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'valid']:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        for split_name, source_dir, target_dir in [
            ('train', dataset_root / 'train' / class_name, output_root / 'train' / class_name),
            ('valid', dataset_root / 'valid' / class_name, output_root / 'valid' / class_name),
        ]:
            if not source_dir.exists():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            image_paths = sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}])
            limit = train_per_class if split_name == 'train' else valid_per_class
            for image_path in image_paths[:limit]:
                shutil.copy2(image_path, target_dir / image_path.name)

    return output_root / 'train', output_root / 'valid'


def main():
    parser = argparse.ArgumentParser(description='Train a CNN on the provided plant disease dataset and run a prediction sample.')
    parser.add_argument('--dataset-root', type=str, default=r'C:\Users\siddh\Downloads\plant dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps-per-epoch', type=int, default=10)
    parser.add_argument('--validation-steps', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-model', type=str, default=r'models\mobilenet_benchmark_final.keras')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / 'train'
    valid_dir = dataset_root / 'valid'
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(f'Expected train/valid dirs under {dataset_root}')

    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])[:5]
    subset_root = Path('models') / 'subset_dataset'
    train_dir, valid_dir = build_subset_dataset(dataset_root, subset_root, class_names)
    print(f'Using subset dataset at {subset_root} with classes: {class_names}')

    train_generator, valid_generator = build_training_generators(
        str(train_dir),
        str(valid_dir),
        image_size=(224, 224),
        batch_size=args.batch_size,
        preprocessing_function=None,
    )

    model = build_baseline_cnn_model(num_classes=len(class_names))
    model = compile_model(model, learning_rate=1e-4)
    print(f'Training on {train_dir} for {args.epochs} epoch(s)...')
    history = train_model(
        model,
        train_generator,
        valid_generator,
        epochs=args.epochs,
        callbacks=None,
        steps_per_epoch=min(args.steps_per_epoch, len(train_generator)),
        validation_steps=min(args.validation_steps, len(valid_generator)),
    )

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f'Model saved to {output_path}')

    probabilities = model.predict(valid_generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=-1)
    y_true = valid_generator.classes
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f'Validation accuracy: {accuracy:.4f}')
    print(f'Validation macro F1: {macro_f1:.4f}')

    sample_image = find_sample_image(valid_dir)
    print(f'Prediction sample: {sample_image}')
    pred_idx, confidence, top_predictions = predict_with_confidence(str(sample_image), model=model, top_k=3)
    print('Top predictions:')
    for label, score in top_predictions:
        print(f'  - {label}: {score:.4f}')
    print('Predicted class:', get_class_name(pred_idx))


if __name__ == '__main__':
    main()
