from time import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Vectorizer import Vectorizer


class TextClassifier:
    def __init__(self):
        #initiate USE
        self.vectorizer = Vectorizer("en")
        #temporary
        self.T = 2
        self.D = 10

    def dataprep(self, data):
        #get a vector
        #vector = vectorizer.get_vector_from_text(sentences)
        data.dropna(how="any", inplace=True, axis=1)
        data.columns = ['label', 'message']

        #Split in sentences
        #data["sentences"] = data.message.apply(lambda x: x.split('.'))#Old Version, but embeddings of various shape
        data["sentences"] = [[x] for x in data.message]
        length = [len(x) for x in data.sentences]

        #Get embeddings
        data["vectors"] = data.sentences.apply(lambda x: self.vectorizer.get_vector_from_text(x))
        print([v.shape for v in data.vectors])

        #map the label to 0 or 1(spam)
        data['b_labels'] = data['label'].map({'ham': 0, 'spam': 1})
        #data['b_labels'] = [0 if x=="ham" else 1 for x in data.label] 
        Y = np.asarray(data['b_labels'].values)


        #split training and test set	
        #df_train, df_test, Ytrain, Ytest = train_test_split(data['vectors'], Y, test_size=0.33)
        return data.vectors, Y
        #return df_train, df_test, Ytrain, Ytest

    def feature_engineering(self):
        #TBD
        pass

    def lstm(self, X, Y):
        print(type(X))
        print(X)
        #lstm(self,df_train, df_test, Ytrain, Ytest):
        #inputs = np.expand_dims(self.X, -1)
        i = Input(shape=(1, 512))
        #x = LSTM(5, return_sequences=True)(i)
        #x = GlobalMaxPooling1D()(x)
        x = Dense(10, activation='relu')(i)
        x = Dense(1, activation='sigmoid')(x)


        model = Model(i, x)
        
        model.compile(
          loss='binary_crossentropy',
          #optimizer='rmsprop',
          #optimizer=Adam(lr=0.01),
          #optimizer=SGD(lr=0.1, momentum=0.9),
          optimizer='adam',
          metrics=['accuracy'],
        )


        # train the RNN
        r = model.fit(
          #df_train, Ytrain,
          X, Y,
          epochs=20,
          #validation_data=(df_test, Ytest),
          validation_split=0.5,
        )

        #Performance evvaluation
        plt.plot(r.history["loss"], label = "loss")
        plt.plot(r.history["val_loss"], label = "val_loss")
        plt.legend()

    def train(self, data):
        X, Y = self.dataprep(data)
        self.lstm(X, Y)
        #df_train, df_test, Ytrain, Ytest = self.dataprep(data)
        #self.lstm(df_train, df_test, Ytrain, Ytest)
    
    def run(self):
        pass

if __name__=="__main__":
    #test uses the "SPAM" dataset from uci as found on Kaggle : https://www.kaggle.com/uciml/sms-spam-collection-dataset
    now = time()
    data = pd.read_csv("spam2.csv", encoding='latin-1')
    data = data.iloc[0:5]
    sentences = ['I prefer Python over Java', 'I like coding in Python', 'coding is fun']
    clf = TextClassifier()
    clf.train(data)

    print((time()-now)/60)

