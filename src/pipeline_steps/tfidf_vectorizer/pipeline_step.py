import click
import numpy as np
import dill
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@click.command()
@click.option('--in-path', default="/mnt/clean_text.data")
@click.option('--out-path', default="/mnt/tfidf_vectors.data")
@click.option('--max-features', default=5000)
@click.option('--ngram-range', default=3)
@click.option('--action', default="train", 
        type=click.Choice(['predict', 'train']))
@click.option('--model-path', default="/mnt/tfidf.model")
def run_pipeline(
        in_path, 
        out_path, 
        max_features,
        ngram_range,
        action,
        model_path):

    with open(in_path, 'rb') as in_f:
        x = dill.load(in_f)
        x = x.flatten()

    if action == "train":
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            preprocessor=lambda x: x, # We're using cleantext
            tokenizer=None, # We're using spacy
            ngram_range=(1, ngram_range))
        print(x)

        y = tfidf_vectorizer.fit_transform(x)
        logging.info('y:')
        logging.info(y)
        logging.info("saving vectorizer",)
        with open(model_path, "wb") as model_f:
            dill.dump(tfidf_vectorizer, model_f)

    elif action == "predict":
        with open(model_path, "rb") as model_f:
            tfidf_vectorizer = dill.load(model_f)

    y = tfidf_vectorizer.transform(x)

    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)

if __name__ == "__main__":
    run_pipeline()