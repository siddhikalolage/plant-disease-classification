from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_augmentation_generator(rotation_range: int = 20,
                                  width_shift_range: float = 0.2,
                                  height_shift_range: float = 0.2,
                                  shear_range: float = 0.2,
                                  zoom_range: float = 0.2,
                                  horizontal_flip: bool = True):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        rescale=1.0/255.0,
    )
