import click
import numpy as np
import dill
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Running for tests
    from .Transformer import Transformer
except:
    # Running from CLI
    from Transformer import Transformer


@click.command()
@click.option('--in-path', default="/mnt/word_index.data")
@click.option('--out-path', default="/mnt/embedded_matrix.data")
@click.option('--glove-file', default="/mnt/glove.42B.300d.txt")
def run_pipeline(
        in_path, 
        out_path, 
        glove_file):

    with open(in_path, 'rb') as in_f:
        x = dill.load(in_f)

    embedder = Transformer()
    y = embedder.predict(x, glove_file)

    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)

if __name__ == "__main__":
    run_pipeline()