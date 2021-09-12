from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy
import skimage.transform
from skimage import img_as_ubyte
import numpy as np

class IdentityBlock(layers.Layer):
    def __init__(self, kernel_size, filters, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters

    def call(self, X):
        F1, F2, F3 = self.filters
        X_shortcut = X

        X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = layers.Conv2D(filters=F2, kernel_size=self.kernel_size, strides=(1, 1), padding="same")(X)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
        X = layers.BatchNormalization(axis=3)(X)

        X = layers.Add()([X, X_shortcut])
        X = layers.Activation("relu")(X)

        return X

    def compute_output_shape(self, input_shape):
        return input_shape

class ConvBlock(layers.Layer):
    def __init__(self, kernel_size, filters, strides = (2, 2), **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides

    def call(self, X):
        F1, F2, F3 = self.filters
        X_shortcut = X

        X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=self.strides, padding="valid")(X)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = layers.Conv2D(filters=F2, kernel_size=self.kernel_size, strides=(1, 1), padding="same")(X)
        X = layers.BatchNormalization(axis=3)(X)
        X = layers.Activation("relu")(X)

        X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid")(X)
        X = layers.BatchNormalization(axis=3)(X)

        X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=self.strides, padding="valid")(X_shortcut)
        X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

        X = layers.Add()([X, X_shortcut])
        X = layers.Activation("relu")(X)

        return X

    def compute_output_shape(self, input_shape):
        return input_shape


class ResNet(models.Sequential):
    name = "ResNet"
    input_shape = (32, 32, 3)

    def __init__(self, classes, optimizer):
        super().__init__()

        self.add(layers.ZeroPadding2D((3, 3)))
        self.add(layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), input_shape=self.input_shape))
        self.add(layers.BatchNormalization(axis=3))
        self.add(layers.Activation('relu'))
        self.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.add(ConvBlock(filters=(64, 64, 256), kernel_size=(3, 3), strides=(1, 1)))
        self.add(IdentityBlock(filters=(64, 64, 256), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(64, 64, 256), kernel_size=(3, 3)))

        self.add(ConvBlock(filters=(128, 128, 512), kernel_size=(3, 3), strides=(2, 2)))
        self.add(IdentityBlock(filters=(128, 128, 512), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(128, 128, 512), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(128, 128, 512), kernel_size=(3, 3)))

        self.add(ConvBlock(filters=(256, 256, 1024), kernel_size=(3, 3), strides=(2, 2)))
        self.add(IdentityBlock(filters=(256, 256, 1024), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(256, 256, 1024), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(256, 256, 1024), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(256, 256, 1024), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(256, 256, 1024), kernel_size=(3, 3)))

        self.add(ConvBlock(filters=(512, 512, 2048), kernel_size=(3, 3), strides=(2, 2)))
        self.add(IdentityBlock(filters=(512, 512, 2048), kernel_size=(3, 3)))
        self.add(IdentityBlock(filters=(512, 512, 2048), kernel_size=(3, 3)))

        self.add(layers.AveragePooling2D(pool_size=(2, 2)))

        self.add(layers.Flatten())
        self.add(layers.Dense(classes, activation='softmax'))

        self.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    def resize_images(self, images):
        return images