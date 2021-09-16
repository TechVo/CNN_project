from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense, Input, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
import skimage.transform
from skimage import img_as_ubyte
import numpy as np

class ResNet50():
    name = "ResNet50"
    input_size = 64

    def identity_block(self, X, f, filters):
        F1, F2, F3 = filters
        X_shortcut = X

        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def convolutional_block(self, X, f, filters, s = 2):
        F1, F2, F3 = filters
        X_shortcut = X

        X = Conv2D(F1, (1, 1), strides = (s,s))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        X = Conv2D(F2, (f, f), strides = (1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)

        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut) 

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    
    def build_model(self, classes, optimizer, filters):
        
        input_shape = (ResNet50.input_size, ResNet50.input_size, filters)
        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(64, (7, 7), strides = (2, 2))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        X = self.convolutional_block(X, 3, [64, 64, 256], 1)
        X = self.identity_block(X, 3, [64, 64, 256])
        X = self.identity_block(X, 3, [64, 64, 256])

        X = self.convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])

        X = self.convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        
        X = self.convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
        X = self.identity_block(X, 3, [512, 512, 2048])
        X = self.identity_block(X, 3, [512, 512, 2048])
        
        X = AveragePooling2D((2, 2))(X)

        X = Flatten()(X)
        X = Dense(classes, activation='softmax')(X)

        model = Model(inputs = X_input, outputs = X)
        model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

        return model
    
    def __init__(self, classes, optimizer, filters = 3):
        self.model = self.build_model(classes, optimizer, filters)

    @staticmethod
    def resize_images(images):
        tmp_images = []
        for image in images:
            image = skimage.transform.resize(image, (ResNet50.input_size, ResNet50.input_size), mode='constant')
            image = img_as_ubyte(image)
            tmp_images.append(image)
        return np.array(tmp_images)