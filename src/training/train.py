from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def build_training_generators(train_dir, valid_dir, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

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


def get_training_callbacks(checkpoint_path="models/plant_disease_cnn_model.keras"):
    return [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
    ]
