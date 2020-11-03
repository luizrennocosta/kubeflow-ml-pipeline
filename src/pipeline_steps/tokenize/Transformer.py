import re
import pandas as pd
import logging
import nltk

nltk.download("stopwords")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Transformer:
    def __init__(self):
        self.tokenizer = None

    def fit(self, X, num_words, max_length):
        logging.warning(X)
        self.tokenizer = Tokenizer(num_words=num_words, lower=False, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X)
        self.max_length = max_length
        self.num_words = num_words

        # logging.warning(X_tokenized)
        return self.tokenizer

    def predict(self, X):
        logging.info("Predicting")
        Y = self.tokenizer.texts_to_sequences(X)
        logging.info("Padding output")
        Y = pad_sequences(
            Y, padding="post", truncating="post", maxlen=self.max_length
        )
        # logging.warning(X_tokenized)
        return Y, self.tokenizer.word_index

    @staticmethod
    def tokenize_text(train_text, num_words, max_length, tokenizer=None):
        if tokenizer == None:
            tokenizer = Tokenizer(num_words=num_words, lower=False, oov_token="<OOV>")
            tokenizer.fit_on_texts(train_text)

        train_sequences = tokenizer.texts_to_sequences(train_text)
        train_padded = pad_sequences(
            train_sequences, padding="post", truncating="post", maxlen=max_length
        )
        return train_padded, tokenizer.word_index, tokenizer