import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path


class Transformer:
    def predict(self, X, path_to_glove_file):
        # logging.info(X)
        embedding_matrix = Transformer.create_embed_matrix(X, path_to_glove_file)
        # logging.info(embedding_matrix)
        return embedding_matrix

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def create_embed_matrix(word_index, path_to_glove_file):
        num_tokens = len(word_index) + 2
        hits = 0
        misses = 0
        embeddings_index = {}
        logging.info("Verifying pre-embedded words")
        with tqdm(total=Path(path_to_glove_file).stat().st_size) as pbar:
            with open(path_to_glove_file) as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    embeddings_index[word] = coefs
                    pbar.update(len(line))

        logging.info(f"coefs shape: {coefs.shape}")
        embedding_dim = coefs.shape[0]
        logging.info("Found %s word vectors." % len(embeddings_index))

        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in tqdm(word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        logging.info("Converted %d words (%d misses)" % (hits, misses))
        return embedding_matrix
