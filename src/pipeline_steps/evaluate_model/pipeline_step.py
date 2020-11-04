import click
import numpy as np
import dill
from sklearn import metrics
from pathlib import Path
import pandas as pd
import json


@click.command()
@click.option("--data-folder", default="/mnt")
@click.option("--predicted-train-data", default="/mnt/data/predicted_train.data")
@click.option("--predicted-val-data", default="/mnt/data/predicted_val.data")
@click.option("--predicted-test-data", default="/mnt/data/predicted_test.data")
def run_pipeline(
    data_folder,
    predicted_train_data,
    predicted_val_data,
    predicted_test_data,
):

    predicted_data = {}
    target_data = {}

    with open(Path(data_folder).joinpath("train.data"), "rb") as train_f:
        target_data["_train"] = dill.load(train_f)

    with open(Path(data_folder).joinpath("val.data"), "rb") as val_f:
        target_data["_val"] = dill.load(val_f)

    with open(Path(data_folder).joinpath("test.data"), "rb") as test_f:
        target_data["_test"] = dill.load(test_f)
        print(target_data["_test"])

    with open(predicted_train_data, "rb") as train_f:
        predicted_data["_train"] = np.argmax(dill.load(train_f), axis=1)

    with open(predicted_val_data, "rb") as val_f:
        predicted_data["_val"] = np.argmax(dill.load(val_f), axis=1)

    with open(predicted_test_data, "rb") as test_f:
        predicted_data["_test"] = np.argmax(dill.load(test_f), axis=1)

    evaluated_metrics = {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
    }

    metadata_json = {}
    metrics_json = {}

    metrics_json["metrics"] = []
    metadata_json["outputs"] = []
    for suffix in ["_train", "_val", "_test"]:
        Y = predicted_data[suffix]
        _, labels = target_data[suffix]
        cm = metrics.confusion_matrix(labels, Y)

        with open(f"/cm{suffix}.csv", "w") as cm_f:
            pd.DataFrame(cm).to_csv(cm_f, header=False, index=False)

        suffix = suffix.replace("_", "-")

        cm_dict = {}
        cm_dict["type"] = "confusion_matrix"
        cm_dict["format"] = "csv"
        cm_dict["schema"] = [
            {"name": "target", "type": "CATEGORY"},
            {"name": "predicted", "type": "CATEGORY"},
        ]
        cm_dict["source"] = f"/cm{suffix}.csv"
        cm_dict["labels"] = labels.flatten().tolist()

        metadata_json["outputs"].append(cm_dict)
        for metric in evaluated_metrics:
            metric_dict = {}
            metric_dict["name"] = metric + suffix
            print(metric)
            if metric == "precision" or metric == "recall":
                metric_dict["numberValue"] = evaluated_metrics[metric](
                    labels, Y, average="macro"
                )
            else:
                metric_dict["numberValue"] = evaluated_metrics[metric](labels, Y)
            metric_dict["format"] = "RAW"
            metrics_json["metrics"].append(metric_dict)

    with open("/mlpipeline-metrics.json", "w") as output_metrics_f:
        json.dump(metrics_json, output_metrics_f)

    with open("/mlpipeline-ui-metadata.json", "w") as output_metadata:
        json.dump(metadata_json, output_metadata)


if __name__ == "__main__":
    run_pipeline()
