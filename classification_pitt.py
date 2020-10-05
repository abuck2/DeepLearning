import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

class classifier:
    def __init__(self):
        print("Tensorflow version : {}".format(tf.__version__))
        self.data = load_breast_cancer()
        #print(self.data)

    def dataprep(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.data.data, self.data.target, test_size = 0.3)
        self.N, self.D = X_train.shape
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)

    def train(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.D,)),
            tf.keras.layers.Dense(1, activation = "sigmoid")
            ])

        model.compile(optimizer = 'adam',
                loss="binary_crossentropy",
                metrics=['accuracy'])
        r = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs = 100)
        
        print("train score : ", model.evaluate(self.X_train, self.y_train))
        print("test score : ", model.evaluate(self.X_test, self.y_test))
        
        plt.plot(r.history["loss"], label="loss")
        plt.plot(r.history["val_loss"], label="val_loss")
        plt.legend()

        plt.plot(r.history["accuracy"], label="accuracy")
        plt.plot(r.history["val_accuracy"], label="val_acc")
        plt.legend()

        plt.show()

    def run(self):
        self.dataprep()
        self.train()

class regressor:
    def __init__(self):
        self.data = pd.read_csv("moore.csv", header = None).values
        self.X = data[:,0].reshape(-1,1)
        self.y = data[:,1]

    def dataprep(self):
        self.y = np.log(self.y)
        self.X  = self.X - self.X.mean()
    
    def train(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1)
            ])

        model.compile(optimizer = tf.keras.optimizers.SGD(0.001, 0.9),
                loss="mse",
                metrics=['accuracy'])
        r = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs = 100)
        
        print("train score : ", model.evaluate(self.X_train, self.y_train))
        print("test score : ", model.evaluate(self.X_test, self.y_test))
        
        plt.plot(r.history["loss"], label="loss")
        plt.plot(r.history["val_loss"], label="val_loss")
        plt.legend()

        plt.plot(r.history["accuracy"], label="accuracy")
        plt.plot(r.history["val_accuracy"], label="val_acc")
        plt.legend()

        plt.show()

    def run(self):
        self.dataprep()
        

if __name__=="__main__":
    #clf = classifier()
    #clf.run()

    reg = regressor()
    reg.run()
