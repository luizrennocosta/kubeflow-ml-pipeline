import click
import dill

try:
    # Running for tests
    from .Transformer import Transformer
except:
    # Running from CLI
    from Transformer import Transformer


@click.command()
@click.option("--in-path", default="/mnt/data/train.data")
@click.option("--out-path", default="/mnt/tokenized_text.data")
@click.option("--word-index-path", default="/mnt/word_index.data")
@click.option("--tokenizer-path", default="/mnt/tokenizer.model")

@click.option('--action', default="train", 
        type=click.Choice(['predict', 'train']))

@click.option("--num-words", default=20000)
@click.option("--max-length", default=300)

def run_pipeline(in_path, out_path, word_index_path, tokenizer_path, action, num_words, max_length):

    with open(in_path, "rb") as in_f:
        x, _ = dill.load(in_f)
        
    x = x.flatten()

    if action =='train':
        keras_tokenizer_transformer = Transformer()
        keras_tokenizer_transformer.fit(x, num_words, max_length)

        y, word_index = keras_tokenizer_transformer.predict(x)

        with open(word_index_path, "wb") as out_f:
            dill.dump(word_index, out_f)

        with open(tokenizer_path, "wb") as transformer_f:
            dill.dump(keras_tokenizer_transformer, transformer_f)

    elif action == 'predict':
        with open(tokenizer_path, "rb") as transformer_f:
            keras_tokenizer_transformer = dill.load(transformer_f)

        y, _ = keras_tokenizer_transformer.predict(x)
    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)

if __name__ == "__main__":
    run_pipeline()