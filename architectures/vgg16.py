from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy

class VGG16(models.Sequential):
    name = "VGG 16"
    
    def __init__(self, input_shape, classes, optimizer):
        super().__init__()
        
        self.add(layers.Conv2D(filters=64 ,kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same'))
        self.add(layers.Conv2D(filters=64 ,kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu' padding='same'))
        self.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu' padding='same'))
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

    def resize_images(images):
        tmp_images = []
        for image in images:
            image = skimage.transform.resize(image, (224, 224), mode='constant')
            image = img_as_ubyte(image)
            tmp_images.append(image)
        return np.array(tmp_images)