import click
import numpy as np
import dill
from sklearn.linear_model import LogisticRegression

from tensorflow.keras import layers
from tensorflow.keras import Input, Model, initializers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from pathlib import Path
def create_model(embedding_layer, n_classes):
    int_sequences_input = Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(int_sequences_input, preds)
    return model


@click.command()
@click.option("--in-path", default="/mnt/tokenized_text.data")
@click.option("--embed-weight", default="/mnt/embedded_matrix.data")
@click.option("--labels-path", default="/mnt/labels.data")
@click.option("--data-folder", default="/mnt")
@click.option("--out-path", default="/mnt/text_prediction.data")

@click.option("--action", default="train", type=click.Choice(["predict", "train"]))
@click.option("--model-path", default="/mnt/deep_text.tf")
@click.option("--epochs", default=20)
@click.option("--batch-size", default=1024)
@click.option("--optimizer", default="rmsprop")
@click.option("--metrics", multiple=True, default=["acc"])
def run_pipeline(
    in_path,
    embed_weight,
    labels_path,
    data_folder,
    out_path,
    action,
    model_path,
    epochs,
    batch_size,
    optimizer,
    metrics,
):

    with open(in_path, "rb") as in_f:
        x = dill.load(in_f)
    with open(embed_weight, "rb") as ew_f:
        embedding_matrix = dill.load(ew_f)
    with open(labels_path, "rb") as f:
        labels = dill.load(f)

    with open(Path(data_folder).joinpath('train.data'), "rb") as train_f:
        (X_train, Y_train) = dill.load(train_f)
    
    with open(Path(data_folder).joinpath('val.data'), "rb") as val_f:
        (X_val, Y_val) = dill.load(val_f)

    with open(Path(data_folder).joinpath('test.data'), "rb") as test_f:
        (X_test, Y_test) = dill.load(test_f)

    opm = RMSprop(learning_rate=0.01)
    print(embedding_matrix.shape)
    if action == "train":
        embedding_layer = layers.Embedding(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            embeddings_initializer=initializers.Constant(embedding_matrix),
            trainable=False,
        )
        deep_model = create_model(embedding_layer, max(labels) + 1)
        deep_model.compile(
            loss="sparse_categorical_crossentropy", optimizer=opm, metrics=metrics
        )
        deep_model.fit(
            X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size
        )

        Y_train_hat = deep_model.predict(X_train)
        Y_val_hat = deep_model.predict(X_val)
        Y_test_hat = deep_model.predict(X_test)

        with open(Path(data_folder).joinpath('predicted_train.data'), "wb") as out_f:
            dill.dump(Y_train_hat, out_f)
        with open(Path(data_folder).joinpath('predicted_val.data'), "wb") as out_f:
            dill.dump(Y_val_hat, out_f)
        with open(Path(data_folder).joinpath('predicted_test.data'), "wb") as out_f:
            dill.dump(Y_test_hat, out_f)

        deep_model.save(model_path)

    elif action == "predict":
        with open(model_path, "rb") as model_f:
            deep_model = dill.load(model_f)

        y = deep_model.predict_proba(x)

        with open(out_path, "wb") as out_f:
            dill.dump(y, out_f)


if __name__ == "__main__":
    run_pipeline()