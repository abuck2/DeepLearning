"""
Small classes with basic Convulotional networks
"""

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalMaxPooling2D,\
        BatchNormalization, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
#import tensorflow as tf

class BasicCNN:
    def __init__(self):
        pass

    def dataprep(self, x_train, y_train, x_test, y_test):
        x_train, x_test = x_train/255.0, x_test/255.0
        y_train, y_test = y_train.flatten(), y_test.flatten()        
        


        self.K = len(set(y_train))
        print("x_train shape = {}".format(x_train.shape))
        print("y_train shape = {}".format(y_train.shape))
        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train, x_test, y_test):
        i = Input(shape = x_train[0].shape)
        """
        #One model
        x = Conv2D(32, (3,3), strides = 2, activation = "relu")(i)
        x = Conv2D(64, (3,3), strides = 2, activation = "relu")(x)
        x = Conv2D(128, (3,3), strides = 2, activation = "relu")(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation = "relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(self.K, activation = "softmax")(x)
        """

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(self.K, activation='softmax')(x)
        model = Model(i, x)
        
        model.compile(optimizer = "adam",
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])
        
        #Data augmentation
        batch_size = 32
        data_generator = ImageDataGenerator(
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            horizontal_flip=True)
        train_generator = data_generator.flow(x_train, y_train, batch_size)
        steps_per_epoch = x_train.shape[0] // batch_size
        r = model.fit_generator(train_generator,
                validation_data=(x_test, y_test), 
                steps_per_epoch=steps_per_epoch,
                epochs=50)
        
        
        
        #r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10)
        
        plt.plot(r.history["loss"], label = "loss")
        plt.plot(r.history["val_loss"], label = "val_loss")
        plt.legend()
        plt.show()
        
        plt.plot(r.history["accuracy"], label = "accuracy")
        plt.plot(r.history["val_accuracy"], label = "val_accuracy")
        plt.legend()
        plt.show()

    def predict(self):
        pass

    def run(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, y_train, x_test, y_test = self.dataprep(x_train, y_train, x_test, y_test)
        self.train(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    nn = BasicCNN()
    nn.run()
