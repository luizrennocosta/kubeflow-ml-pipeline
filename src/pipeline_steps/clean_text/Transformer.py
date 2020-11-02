import re
import pandas as pd
import logging
import nltk
nltk.download('stopwords')

class Transformer():

    def predict(self, X, feature_names=['review_body']):
        logging.warning(X)
        X_clean = Transformer.transform_clean_text(X, feature_names)
        logging.warning(X_clean)
        return X_clean

    def fit(self, X, y=None, **fit_params):
        return self
    
    @staticmethod
    def transform_clean_text(text_df, features):

        stemmer = nltk.stem.SnowballStemmer("english")
        STOPWORDS = set(nltk.corpus.stopwords.words('english'))
        for col in features:
            text_df[col] = text_df[col].astype(str)
            text_df[col] = text_df[col].str.lower()
            text_df[col] = text_df[col].str.replace('[!"#$%&()*\'+,-./:;<=>?@[\\]^_{|}~`\t\n0123456789]', ' ')
            text_df[col] = text_df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))
        return text_df.to_numpy()