"""
Small classes with basic feed-forward networks to use as a baseline on mnist
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MnistANN:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        

    def dataprep(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
        self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0

    def train(self):
        
        """
        Flatten to flatten the images as one vector per image as input
        Dropout to avoid having the network relying on one input too strongly (basic regul)
        final layer : 10 categories, softmax to multi class
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(100, activation = "relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation = "softmax")
            ])

        model.compile(optimizer = "adam",
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])

        model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test),
                epochs = 5)
        """
        plt.plot(r.history["loss"], label = "loss")
        plt.plot(r.history["val_loss"], label = "val_loss")
        plt.legend()
        plt.show()
        
        plt.plot(r.history["accuracy"], label = "acc")
        plt.plot(r.history["val_accuracy"], label = "val_acc")
        plt.legend()
        plt.show()

        """
        print(model.evaluate(self.x_test, self.y_test))
    


    def run(self):
        self.dataprep()
        self.train()

if __name__ == "__main__":
    nn = MnistANN()
    nn.run()
