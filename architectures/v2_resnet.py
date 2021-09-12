from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy

def identity_block(X, f, filters):
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

def convolutional_block(X, f, filters, s = 2):
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

def ResNet(classes, optimizer, input_shape=(32, 32, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(32, (8, 8), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [32, 32, 128], s = 1)
    X = identity_block(X, 3, [32, 32, 128])
    X = identity_block(X, 3, [32, 32, 128])
    X = MaxPooling2D((4, 4), strides=(2, 2), padding="same")(X)

    X = convolutional_block(X, f = 3, filters = [16, 16, 256], s = 2)
    X = identity_block(X, 3, [16, 16, 256])
    X = identity_block(X, 3, [16, 16, 256])
    X = identity_block(X, 3, [16, 16, 256])
    X = identity_block(X, 3, [16, 16, 256])
    X = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(X)

    X = convolutional_block(X, f = 3, filters = [8, 8, 512], s = 2)
    X = identity_block(X, 3, [8, 8, 512])
    X = identity_block(X, 3, [8, 8, 512])
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)
    
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    return model
