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
@click.option("--num-words", default=1000)
@click.option("--max-length", default=150)
def run_pipeline(in_path, out_path, num_words, max_length):

    keras_tokenizer_transformer = Transformer()
    with open(in_path, "rb") as in_f:
        x = dill.load(in_f)

    y = keras_tokenizer_transformer.predict(x, num_words, max_length)
    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)


if __name__ == "__main__":
    run_pipeline()