"""
Training utilities for Plant Disease Classification.

Supported architectures:
    - Baseline CNN
    - EfficientNetB0
    - MobileNetV3Small / MobileNetV3Large

The module provides:
    - Dataset generators
    - Model builders
    - Model compilation
    - Training callbacks
    - Two-stage transfer learning
    - Safe fine-tuning of pretrained backbones
"""

from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------
# DEFAULT CONSTANTS
# ---------------------------------------------------------------------

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_NUM_CLASSES = 38


# ---------------------------------------------------------------------
# DATA GENERATORS
# ---------------------------------------------------------------------

def build_training_generators(
    train_dir: str,
    valid_dir: str,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
):
    """
    Build training and validation ImageDataGenerator objects.

    Images are rescaled to [0, 1].

    Returns:
        train_generator
        valid_generator
    """

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    return train_generator, valid_generator


# ---------------------------------------------------------------------
# BASELINE CNN
# ---------------------------------------------------------------------

def build_baseline_cnn_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5,
):
    """
    Build a lightweight CNN baseline.
    """

    inputs = keras.Input(shape=input_shape, name="image")

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
    )(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="baseline_cnn",
    )

    return model


# ---------------------------------------------------------------------
# EFFICIENTNETB0
# ---------------------------------------------------------------------

def build_efficientnetb0_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    trainable: bool = False,
):
    """
    Build an EfficientNetB0 transfer-learning classifier.

    The backbone is frozen by default.
    """

    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    backbone.trainable = trainable

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    x = backbone(inputs, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="efficientnetb0_classifier",
    )

    return model


# ---------------------------------------------------------------------
# MOBILENETV3
# ---------------------------------------------------------------------

def build_mobilenetv3_model(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    trainable: bool = False,
    model_type: str = "small",
):
    """
    Build a MobileNetV3 transfer-learning classifier.

    Args:
        num_classes:
            Number of output classes.

        input_shape:
            Input image shape.

        dropout_rate:
            Dropout before classifier.

        trainable:
            Whether the backbone starts trainable.

        model_type:
            "small" or "large".
    """

    if model_type.lower() == "small":
        backbone = keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    elif model_type.lower() == "large":
        backbone = keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        raise ValueError(
            "model_type must be either 'small' or 'large'."
        )

    backbone.trainable = trainable

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    # IMPORTANT:
    # Always call the backbone here.
    #
    # During the initial frozen stage we explicitly use
    # training=False so BatchNormalization statistics do not
    # change unexpectedly.
    x = backbone(
        inputs,
        training=False,
    )

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="dense",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"mobilenetv3_{model_type.lower()}_classifier",
    )

    return model


# ---------------------------------------------------------------------
# BACKBONE DISCOVERY
# ---------------------------------------------------------------------

def get_backbone(model: keras.Model) -> keras.Model:
    """
    Locate the pretrained backbone inside a transfer-learning model.

    The backbone is the nested Functional model.

    This avoids accidentally modifying the outer classifier model.
    """

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer

    raise ValueError(
        "Could not find a nested pretrained backbone in the model."
    )


# ---------------------------------------------------------------------
# FINE-TUNING
# ---------------------------------------------------------------------

def set_fine_tune_layers(
    model: keras.Model,
    num_layers: int,
):
    """
    Safely unfreeze the final `num_layers` layers of the pretrained
    backbone.

    IMPORTANT:
        The outer classifier remains intact.

        We modify:
            model.layers -> nested backbone -> backbone.layers

        We do NOT replace model.output.

    BatchNormalization layers remain frozen because updating their
    moving statistics during small fine-tuning runs can destabilize
    transfer learning.

    Returns:
        The pretrained backbone.
    """

    if num_layers <= 0:
        raise ValueError(
            "num_layers must be greater than zero."
        )

    backbone = get_backbone(model)

    # First freeze everything.
    backbone.trainable = False

    for layer in backbone.layers:
        layer.trainable = False

    # Determine how many actual backbone layers can be unfrozen.
    total_layers = len(backbone.layers)

    num_layers = min(
        num_layers,
        total_layers,
    )

    start_index = total_layers - num_layers

    for layer in backbone.layers[start_index:]:
        # Keep BatchNormalization frozen.
        if isinstance(
            layer,
            (
                layers.BatchNormalization,
            ),
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    # IMPORTANT:
    # The outer model must remain trainable.
    model.trainable = True

    return backbone


# ---------------------------------------------------------------------
# BACKWARD-COMPATIBILITY ALIAS
# ---------------------------------------------------------------------

def unfreeze_backbone_layers(
    model: keras.Model,
    num_layers: int,
):
    """
    Backward-compatible alias for set_fine_tune_layers().
    """

    return set_fine_tune_layers(
        model,
        num_layers,
    )


# ---------------------------------------------------------------------
# COMPILE
# ---------------------------------------------------------------------

def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
):
    """
    Compile the classification model.

    Sparse categorical cross entropy is used because the generators
    return integer class labels.
    """

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


# ---------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------

def get_training_callbacks(
    checkpoint_path: str,
):
    """
    Build callbacks used during training.
    """

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=0,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr,
    ]


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_model(
    model: keras.Model,
    train_generator,
    valid_generator,
    epochs: int,
    callbacks=None,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
):
    """
    Train the model using directory generators.
    """

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks or [],
        verbose=1,
    )

    return history


# ---------------------------------------------------------------------
# MODEL VALIDATION HELPERS
# ---------------------------------------------------------------------

def verify_classifier_output(
    model: keras.Model,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """
    Verify that the final model still produces a class probability
    vector rather than a convolutional feature map.

    Raises:
        ValueError if the output shape or activation is incorrect.
    """

    expected_shape = (None, num_classes)

    if model.output_shape != expected_shape:
        raise ValueError(
            "Invalid classifier output. "
            f"Expected {expected_shape}, "
            f"received {model.output_shape}."
        )

    output_layer = model.layers[-1]

    if not isinstance(
        output_layer,
        layers.Dense,
    ):
        raise ValueError(
            "The final model layer must be Dense."
        )

    if output_layer.units != num_classes:
        raise ValueError(
            f"Expected {num_classes} output units, "
            f"received {output_layer.units}."
        )

    return True


# ---------------------------------------------------------------------
# FINE-TUNING VALIDATION
# ---------------------------------------------------------------------

def verify_fine_tuning_state(
    model: keras.Model,
):
    """
    Verify that fine-tuning did not replace the classifier output.

    Returns a dictionary useful for diagnostics.
    """

    backbone = get_backbone(model)

    trainable_backbone_layers = sum(
        1
        for layer in backbone.layers
        if layer.trainable
    )

    non_trainable_backbone_layers = sum(
        1
        for layer in backbone.layers
        if not layer.trainable
    )

    result = {
        "model_output_shape": model.output_shape,
        "backbone_layers": len(backbone.layers),
        "trainable_backbone_layers": trainable_backbone_layers,
        "non_trainable_backbone_layers": non_trainable_backbone_layers,
        "total_parameters": model.count_params(),
    }

    verify_classifier_output(
        model,
        num_classes=model.output_shape[-1],
    )

    return result