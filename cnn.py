"""
Small classes with basic Convulotional networks
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.datasets import fashion_mnist
#import tensorflow as tf

class BasicCNN:
    def __init__(self):
        pass 

    def dataprep(self, data):
        (x_train, y_train), (x_test, y_test) = data.load_data() 
        x_train, x_test = x_train/255.0, x_test/255.0
        
        #To have three dimensional inputs HxWxC
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        #different classes
        self.K = len(set(y_train))
        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train, x_test, y_test):
        i = Input(x_train[0].shape)
        #Args for Conv2D are output feature map (and so filters/kernels), filter dimensions, strides, act fnc, and padding (valid = no padding, same = keep the dimensions)
        x = Conv2D(32, (3,3), strides = 2, activation = "relu", padding = "same")(i)
        x = Conv2D(64, (3,3), strides = 2, activation = "relu", padding = "same")(x)
        x = Conv2D(128, (3,3), strides = 2, activation = "relu", padding = "same")(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation = "relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(self.K, activation = "softmax")(x)

        model = Model(i, x)

        model.compile(optimizer = "adam",
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])

        r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 10)#only 10 because it's done on my laptop...
        plt.plot(r.history['loss'], label = "loss")
        plt.plot(r.history['val_loss'], label = "val_loss")
        plt.legend()
        plt.show()

    def predict(self):
        pass


    def run(self, dataset = "mnist"):
        data = fashion_mnist
        
        x_train, y_train, x_test, y_test =  self.dataprep(data)
        
        self.train(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    
    nn = BasicCNN()
    nn.run()
