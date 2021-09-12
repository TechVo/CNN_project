from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import np_utils

class Datasets():
    def normalize(self, X_train, X_test):
        # Normalize image vectors
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train/255.
        X_test = X_test/255.
        
        return X_train, X_test
    
    def categorize(self, Y_train, Y_test):
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)

        return Y_train, Y_test

    def load_cifar10(self):
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        
        X_train, X_test = self.normalize(X_train, X_test)
        Y_train, Y_test = self.categorize(Y_train, Y_test)
        
        return X_train, Y_train, X_test, Y_test

    def load_cifar100():
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
        
        X_train, X_test = normalize(X_train, X_test)
        Y_train, Y_test = categorize(Y_train, Y_test)
        
        return X_train, Y_train, X_test, Y_test
        
    def load_mnist():
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        
        X_train, X_test = normalize(X_train, X_test)
        Y_train, Y_test = categorize(Y_train, Y_test)
        
        return X_train, Y_train, X_test, Y_test

    def load_fashion_mnist():
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        
        X_train, X_test = normalize(X_train, X_test)
        Y_train, Y_test = categorize(Y_train, Y_test)
        
        return X_train, Y_train, X_test, Y_test