import click
import numpy as np
import dill
from sklearn.linear_model import LogisticRegression
from pathlib import Path


@click.command()
@click.option("--in-path", default="/mnt/tfidf_vectors.data")
@click.option("--labels-path", default="/mnt/labels.data")
@click.option("--out-path", default="/mnt/lr_prediction.data")
@click.option("--c-param", default=0.1)
@click.option("--action", default="train", type=click.Choice(["predict", "train"]))
@click.option("--model-path", default="/mnt/model/lr_text.model")
def run_pipeline(in_path, labels_path, out_path, c_param, action, model_path):

    with open(in_path, "rb") as in_f:
        x, labels = dill.load(in_f)

    if action == "train":
        lr_model = LogisticRegression(C=0.1, solver="sag")

        # with open(labels_path, "rb") as f:
        #     labels = dill.load(f)

        lr_model.fit(x, labels)

        with open(model_path, "wb") as model_f:
            dill.dump(lr_model, model_f)

    elif action == "predict":
        with open(model_path, "rb") as model_f:
            lr_model = dill.load(model_f)

    y = lr_model.predict_proba(x)

    if not Path(out_path).parents[0].exists():
        Path(out_path).parents[0].mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)


if __name__ == "__main__":
    run_pipeline()
