import click
import dill
import pandas as pd
import urllib.requests2i build . seldonio/seldon-core-s2i-python3s2i build . seldonio/seldon-core-s2i-python3
import shutil
import zipfile
from sklearn.preprocessing import LabelEncoder

@click.command()
@click.option('--labels-path', default="/mnt/labels.data")
@click.option('--features-path', default="/mnt/features.data")
@click.option('--csv-url', default="https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_UK_v1_00.tsv.gz")
@click.option('--csv-compression', default="gzip")
@click.option('--csv-separator', default="\t")
@click.option('--column-names', multiple=True, default=['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date'])
@click.option('--features-column', default="review_body")
@click.option('--labels-column', default="product_category")
def run_pipeline(
        labels_path, 
        features_path,
        csv_url, 
        csv_compression,
        csv_separator,
        column_names,
        features_column,
        labels_column):

    # Downloading and saving data
    df = pd.read_csv(csv_url, compression=csv_compression, 
    sep=csv_separator, error_bad_lines=False, names=column_names)

    x = df[[features_column]]

    with open(features_path, "wb") as out_f:
        dill.dump(x, out_f)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[labels_column])

    with open(labels_path, "wb") as out_f:
        dill.dump(y, out_f)

if __name__ == "__main__":
    run_pipeline()