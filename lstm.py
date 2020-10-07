from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RNNBuilder:
    """
    A class to play a little with LSTM, SimpleRNN and GRU layers
    """
    
    def __init__(self, type_nn:str = "rnn", long_dist:bool = True, seq_length:int = 10):
        if type_nn.lower() in ['rnn', 'lstm', 'gru']:
            self.type_nn = type_nn.lower()
        else : 
            raise ValueError("Wrong RNN type. Type must be 'rnn' (simple), 'lstm' or 'gru'")
        self.long_dist = long_dist

        self.T = seq_length

    def get_label(self, x, i1, i2, i3):
        
        """
        x = sequence
        Catégory depends on the sign of past value, in a XOR fashion.
        We can choose if it depends on recent or old values
        """

        if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
            return 1
        if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
            return 1
        if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
            return 1
        if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
            return 1
        return 0

    def dataprep(self):
        #Generate the data
        T = self.T
        self.D = 1
        X = []
        Y = []
        
        
        for t in range(5000):
            x = np.random.randn(T)
            X.append(x)
            
            if self.long_dist :  
                y = self.get_label(x, 0, 1, 2) # long distance : depends on sig value on long term data
            else :
                y = get_label(x, -1, -2, -3) # short distance
            Y.append(y)

        self.X = np.array(X)
        self.Y = np.array(Y)
        self.N = len(X)
        self.T = T
        print("X.shape", self.X.shape, "Y.shape", self.Y.shape)


    def train(self):
        # Now try a simple RNN
        inputs = np.expand_dims(self.X, -1)
        i = Input(shape=(self.T, self.D))

        # make the RNN
        if  self.type_nn == "rnn" :
            x = SimpleRNN(5)(i)
        elif self.type_nn == "lstm" : 
            x = LSTM(5)(i)
        else : 
            x = GRU(5)(i)

        x = Dense(1, activation='sigmoid')(x)
        
        model = Model(i, x)
        model.compile(
          loss='binary_crossentropy',
          #optimizer='rmsprop',
          #optimizer='adam',
          optimizer=Adam(lr=0.01),
          #optimizer=SGD(lr=0.1, momentum=0.9),
          metrics=['accuracy'],
        )


        # train the RNN
        r = model.fit(
          inputs, self.Y,
          epochs=20,
          validation_split=0.5,
        )

	#Performance evvaluation
        plt.plot(r.history["loss"], label = "loss")
        plt.plot(r.history["val_loss"], label = "val_loss")
        plt.legend()
        plt.show()
    
        #forecast using predictions
        validation_target = self.Y[-self.N//2:]
        validation_predictions = []

        # first validation input
        last_x = self.X[-self.N//2] # 1-D array of length T

        while len(validation_predictions) < len(validation_target):
            p = model.predict(last_x.reshape(1, -1, 1))[0,0] # 1x1 array -> scalar

            # update the predictions list
            validation_predictions.append(p)

            # make the new input
            last_x = np.roll(last_x, -1)
            last_x[-1] = p

        plt.plot(validation_target, label='forecast target')
        plt.plot(validation_predictions, label='forecast prediction')
        plt.legend()



    def run(self):
        self.dataprep()
        self.train()



if __name__=="__main__":
    rnn_model = RNNBuilder()
    rnn_model.run()

