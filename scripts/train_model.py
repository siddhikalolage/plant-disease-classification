from pathlib import Path
import argparse

from src.constants import NUM_CLASSES

from src.training.train import (
    build_training_generators,
    build_baseline_cnn_model,
    build_efficientnetb0_model,
    build_mobilenetv3_model,
    compile_model,
    get_training_callbacks,
    train_model,
    set_fine_tune_layers,
)


def build_model(model_name: str):
    """Create the requested model architecture."""

    if model_name == "baseline":
        return build_baseline_cnn_model(
            num_classes=NUM_CLASSES,
            input_shape=(224, 224, 3),
            dropout_rate=0.5,
        )

    if model_name == "efficientnet":
        return build_efficientnetb0_model(
            num_classes=NUM_CLASSES,
            input_shape=(224, 224, 3),
            dropout_rate=0.3,
            trainable=False,
        )

    if model_name == "mobilenet":
        return build_mobilenetv3_model(
            num_classes=NUM_CLASSES,
            input_shape=(224, 224, 3),
            dropout_rate=0.3,
            trainable=False,
            model_type="small",
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a plant disease classification model."
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing train/ and valid/ directories.",
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of stage-1 training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Stage-1 Adam learning rate.",
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "efficientnet",
            "mobilenet",
        ],
        default="baseline",
        help="Model architecture to train.",
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/baseline_cnn.keras"),
        help="Path to save the final trained model.",
    )

    # --------------------------------------------------------
    # STEPS
    # --------------------------------------------------------

    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional number of training batches per epoch.",
    )

    parser.add_argument(
        "--validation-steps",
        type=int,
        default=None,
        help="Optional number of validation batches per epoch.",
    )

    # --------------------------------------------------------
    # FINE TUNING
    # --------------------------------------------------------

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help=(
            "Enable stage-2 fine tuning for pretrained models "
            "after stage-1 training."
        ),
    )

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Number of stage-2 fine-tuning epochs.",
    )

    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
        help="Stage-2 fine-tuning learning rate.",
    )

    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=30,
        help="Number of final backbone layers to unfreeze.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    train_dir = args.dataset_root / "train"
    valid_dir = args.dataset_root / "valid"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {valid_dir}"
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print("=" * 60)
    print("PLANT DISEASE MODEL TRAINING")
    print("=" * 60)

    print(f"Dataset:        {args.dataset_root}")
    print(f"Classes:        {NUM_CLASSES}")
    print(f"Model:          {args.model}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Fine tuning:    {args.fine_tune}")

    if args.fine_tune:
        print(f"FT epochs:      {args.fine_tune_epochs}")
        print(f"FT LR:          {args.fine_tune_learning_rate}")
        print(f"FT layers:      {args.fine_tune_layers}")

    print(f"Output:         {args.output}")

    print("=" * 60)

    # --------------------------------------------------------
    # DATA GENERATORS
    # --------------------------------------------------------

    train_generator, valid_generator = build_training_generators(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        image_size=(224, 224),
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # CLASS VALIDATION
    # --------------------------------------------------------

    if train_generator.num_classes != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} training classes, "
            f"but found {train_generator.num_classes}."
        )

    if valid_generator.num_classes != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} validation classes, "
            f"but found {valid_generator.num_classes}."
        )

    if train_generator.class_indices != valid_generator.class_indices:
        raise ValueError(
            "Training and validation class mappings do not match."
        )

    print("\nClass mapping verified.")

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model(args.model)

    print("\nModel created.")
    print(f"Total parameters: {model.count_params():,}")

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 1: CLASSIFIER TRAINING")
    print("=" * 60)

    model = compile_model(
        model,
        learning_rate=args.learning_rate,
    )

    callbacks = get_training_callbacks(
        checkpoint_path=str(args.output)
    )

    print("\nStarting stage-1 training...\n")

    history_stage1 = train_model(
        model=model,
        train_generator=train_generator,
        valid_generator=valid_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )

    val_accuracy_stage1 = history_stage1.history.get(
        "val_accuracy",
        [],
    )

    if val_accuracy_stage1:
        best_stage1 = max(val_accuracy_stage1)

        print(
            "\nBest stage-1 validation accuracy: "
            f"{best_stage1:.4f}"
        )

    # --------------------------------------------------------
    # STAGE 2: FINE TUNING
    # --------------------------------------------------------

    if args.fine_tune:

        if args.model == "baseline":
            print(
                "\nFine tuning requested, but the baseline CNN "
                "does not contain a pretrained backbone."
            )
            print("Skipping stage 2.")

        else:

            print("\n" + "=" * 60)
            print("STAGE 2: FINE TUNING")
            print("=" * 60)

            print(
                f"Unfreezing last "
                f"{args.fine_tune_layers} backbone layers."
            )

            set_fine_tune_layers(
                model,
                num_layers=args.fine_tune_layers,
                )

            trainable_count = sum(
                1
                for layer in model.layers
                if layer.trainable
            )

            print(
                f"Trainable top-level layers: "
                f"{trainable_count}"
            )

            # Safety check: stage-2 must keep the outer classifier.
            if model.output_shape != (None, NUM_CLASSES):
                raise ValueError(
                    "Fine-tuning changed the classifier output. "
                    f"Expected {(None, NUM_CLASSES)}, "
                    f"received {model.output_shape}."
                )

            # Recompile with a much smaller learning rate.
            model = compile_model(
                model,
                learning_rate=args.fine_tune_learning_rate,
            )

            # Save fine-tuned model to the requested output.
            callbacks_stage2 = get_training_callbacks(
                checkpoint_path=str(args.output)
            )

            print(
                "\nStarting stage-2 fine-tuning...\n"
            )

            history_stage2 = train_model(
                model=model,
                train_generator=train_generator,
                valid_generator=valid_generator,
                epochs=args.fine_tune_epochs,
                callbacks=callbacks_stage2,
                steps_per_epoch=args.steps_per_epoch,
                validation_steps=args.validation_steps,
            )

            val_accuracy_stage2 = history_stage2.history.get(
                "val_accuracy",
                [],
            )

            if val_accuracy_stage2:
                best_stage2 = max(val_accuracy_stage2)

                print(
                    "\nBest stage-2 validation accuracy: "
                    f"{best_stage2:.4f}"
                )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Final model saved to: {args.output}"
    )


if __name__ == "__main__":
    main()