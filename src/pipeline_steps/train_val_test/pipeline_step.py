import click
import numpy as np
import dill
from sklearn.model_selection import train_test_split
from pathlib import Path


@click.command()
@click.option("--in-path", default="/mnt/clean_text.data")
@click.option("--labels-path", default="/mnt/labels.data")
@click.option("--out-folder", default="/mnt/data")
@click.option("--train-ratio", default=0.98)
@click.option("--validation-ratio", default=0.01)
@click.option("--test-ratio", default=0.01)
def run_pipeline(
    in_path,
    labels_path,
    out_folder,
    train_ratio,
    validation_ratio,
    test_ratio,
):

    with open(in_path, "rb") as in_f:
        x = dill.load(in_f)

    with open(labels_path, "rb") as f:
        labels = dill.load(f)

    train_ratio = 0.98
    validation_ratio = 0.01
    test_ratio = 0.01

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, labels, test_size=1 - train_ratio
    )

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    with open(Path(out_folder).joinpath("train.data"), "wb") as out_f:
        dill.dump((X_train, Y_train), out_f)

    with open(Path(out_folder).joinpath("val.data"), "wb") as out_f:
        dill.dump((X_val, Y_val), out_f)

    with open(Path(out_folder).joinpath("test.data"), "wb") as out_f:
        dill.dump((X_test, Y_test), out_f)


if __name__ == "__main__":
    run_pipeline()
