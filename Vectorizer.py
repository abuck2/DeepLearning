import tokenization
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text

class Vectorizer():
    """
    input : list of string
    output : TF tensor
    pretty simple
    model : USE
    """
    def __init__(self, language:str):
        print("Tensorflow Version: ", tf.__version__)
        print("Eager mode: ", tf.executing_eagerly())
        print("Hub version: ", hub.__version__)
        print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
        if language.lower() == "en":
            model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        elif language.lower() == "multi":
            model_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
        else : 
            #Or fallback on multi
            raise ValueError("Language not supported")
        
        self.bert_module = hub.load(model_url)

    def get_vector_from_text(self, text:str):
        # Compute embeddings.
        embeddings = self.bert_module(text)
        return embeddings


if __name__=="__main__":
    vectorizer = Vectorizer("en")
    sentences = ['I prefer Python over Java', 'I like coding in Python', 'coding is fun']
    vector = vectorizer.get_vector_from_text(sentences)
    print(vector)
