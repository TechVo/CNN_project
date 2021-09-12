from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy
import skimage.transform
from skimage import img_as_ubyte
import numpy as np

class VGG16(models.Sequential):
    name = "VGG16"
    input_shape = (224, 224, 3)

    def __init__(self, classes, optimizer):
        super().__init__()
            
        self.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=self.input_shape, padding='same'))
        self.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        
        self.add(layers.Flatten())
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(classes, activation='softmax'))
        
        self.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    def resize_images(self, images):
        tmp_images = []
        x, y, _ = input_shape
        for image in images:
            image = skimage.transform.resize(image, (x, y), mode='constant')
            image = img_as_ubyte(image)
            tmp_images.append(image)
        return np.array(tmp_images)