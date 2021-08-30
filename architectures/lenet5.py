from keras import layers, models
from keras.losses import categorical_crossentropy

class LeNet5(models.Sequential):
    name = "LeNet5"
    
    def __init__(self, input_shape, classes, optimizer):
        super().__init__()
        
        self.add(layers.Conv2D(filters=6 ,kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding='same'))
        self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        self.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid' ))
        self.add(layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(layers.Flatten())
        self.add(layers.Dense(84, activation='tanh'))
        self.add(layers.Dense(classes, activation='softmax'))

        self.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])
