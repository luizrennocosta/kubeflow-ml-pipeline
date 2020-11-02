import re
import pandas as pd
import logging
import nltk

nltk.download("stopwords")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Transformer:
    def predict(self, X, num_words, max_length):
        logging.warning(X)
        X_tokenized = Transformer.tokenize_text(X, num_words, max_length)
        logging.warning(X_tokenized)
        return X_tokenized

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def tokenize_text(train_text, num_words, max_length):

        tokenizer = Tokenizer(num_words=num_words, lower=False, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_text)

        train_sequences = tokenizer.texts_to_sequences(train_text)
        train_padded = pad_sequences(
            train_sequences, padding="post", truncating="post", maxlen=max_length
        )
        return train_padded, tokenizer.word_index