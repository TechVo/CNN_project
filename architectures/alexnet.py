#from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy
import skimage.transform
from skimage import img_as_ubyte
import numpy as np


class AlexNet():
    name = "AlexNet"

    def __init__(self, classes, optimizer):
        self.model = buildModel(classes, optimizer)

    def buildModel(self, classes, optimizer):
        model = models.Sequential()
        
        input_shape = (227, 227, 3)
        model.add(layers.ZeroPadding2D(padding=2))
        model.add(layers.Conv2D(filters=96 ,kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=model.input_shape, padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid' ))
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid' ))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(classes, activation='softmax'))
        
        model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

        return model

    def resize_images(self, images):
        tmp_images = []
        x, y, _ = self.input_shape
        for image in images:
            image = skimage.transform.resize(image, (x, y), mode='constant')
            image = img_as_ubyte(image)
            tmp_images.append(image)
        return np.array(tmp_images)


