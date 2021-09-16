from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy
import skimage.transform
from skimage import img_as_float
import numpy as np

class LeNet5():
    name = "LeNet5"
    
    input_size = 32
    
    def build_model(self, classes, optimizer, filters):
        
        input_shape = (LeNet5.input_size, LeNet5.input_size, filters)
        
        model = models.Sequential()
        
        model.add(layers.Conv2D(filters=6 ,kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding='valid'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid' ))
        model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(classes, activation='softmax'))

        model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])
        
        return model
    
    def __init__(self, classes, optimizer, filters = 3):
        self.model = self.build_model(classes, optimizer, filters)

    @staticmethod
    def resize_images(images):
        tmp_images = []
        for image in images:
            image = skimage.transform.resize(image, (LeNet5.input_size, LeNet5.input_size), mode='constant')
            tmp_images.append(image)
        return np.array(tmp_images)
        

        
        