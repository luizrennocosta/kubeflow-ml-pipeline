import click
import dill

try:
    # Running for tests
    from .Transformer import Transformer
except:
    # Running from CLI
    from Transformer import Transformer


@click.command()
@click.option("--in-path", default="/mnt/clean_text.data")
@click.option("--out-path", default="/mnt/tokenized_text.data")
@click.option("--word-index-path", default="/mnt/word_index.data")
@click.option("--num-words", default=20000)
@click.option("--max-length", default=300)

def run_pipeline(in_path, out_path, word_index_path, num_words, max_length):

    keras_tokenizer_transformer = Transformer()

    with open(in_path, "rb") as in_f:
        x = dill.load(in_f)

    y, word_index = keras_tokenizer_transformer.predict(x, num_words, max_length)
    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)
    with open(word_index_path, "wb") as out_f:
        dill.dump(word_index, out_f)

if __name__ == "__main__":
    run_pipeline()