from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy

class AlexNet(models.Sequential):
    name = "AlexNet"
    
    def __init__(self, input_shape, classes, optimizer):
        super().__init__()
        
        self.add(layers.ZeroPadding2D(padding=2))
        self.add(layers.Conv2D(filters=96 ,kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, padding='valid'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu' padding='valid'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid' ))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
        self.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid' ))

        self.add(layers.Flatten())
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(4096, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(classes, activation='softmax'))
        
        self.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    def convert_to_imagenet_size(images):
        tmp_images = []
        for image in images:
            image = skimage.transform.resize(image,(227,227),mode='constant')
            image = img_as_ubyte(image)
            tmp_images.append(image)
        return np.array(tmp_images)