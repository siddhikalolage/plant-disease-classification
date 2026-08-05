from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from src.constants import NUM_CLASSES


def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
