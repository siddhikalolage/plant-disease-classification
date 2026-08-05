import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small, MobileNetV3Large

from src.constants import NUM_CLASSES


def build_training_generators(train_dir, valid_dir, image_size=(224, 224), batch_size=32, preprocessing_function=None):
    datagen_kwargs = {'preprocessing_function': preprocessing_function} if preprocessing_function is not None else {'rescale': 1.0 / 255.0}
    train_datagen = ImageDataGenerator(**datagen_kwargs)
    valid_datagen = ImageDataGenerator(**datagen_kwargs)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse"
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse"
    )

    return train_generator, valid_generator


def build_baseline_cnn_model(num_classes: int = NUM_CLASSES, input_shape=(224, 224, 3), dropout_rate: float = 0.5):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def build_efficientnetb0_model(num_classes: int = NUM_CLASSES, input_shape=(224, 224, 3), dropout_rate: float = 0.3, trainable: bool = False):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = trainable

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def build_mobilenetv3_model(num_classes: int = NUM_CLASSES, input_shape=(224, 224, 3), dropout_rate: float = 0.3, trainable: bool = False, model_type: str = 'small'):
    model_type = model_type.lower()
    if model_type == 'small':
        base_model = MobileNetV3Small(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
        )
    else:
        base_model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
        )

    base_model.trainable = trainable
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def get_training_callbacks(checkpoint_path="models/plant_disease_cnn_model.keras"):
    return [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
    ]


def train_model(model: tf.keras.Model, train_generator, valid_generator, epochs: int = 20, callbacks=None):
    if callbacks is None:
        callbacks = get_training_callbacks()

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def set_fine_tune_layers(model: tf.keras.Model, trainable_layers: int = 30):
    if not model.layers:
        raise ValueError('Model has no layers to fine tune.')

    base_model = model.layers[0]
    if not isinstance(base_model, tf.keras.Model):
        raise ValueError('Expected the first model layer to be a base pretrained model.')

    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    return model
