from keras.datasets import cifar10

from keras.utils import np_utils

class Datasets():
    def load_cifar10():
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        
        # Normalize image vectors
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train/255.
        X_test = X_test/255.

        # Categorize the data
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)
        
        return X_train, Y_train, X_test, Y_test

