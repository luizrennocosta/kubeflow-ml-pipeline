import click
import urllib.request
import shutil
import zipfile
import os
import logging
import sys
import requests
from tqdm import tqdm

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@click.command()
@click.option("--url", default="http://nlp.stanford.edu/data/glove.42B.300d.zip")
@click.option("--embedding-weights", default="glove.42B.300d.txt")
def run_pipeline(url, embedding_weights):

    # Downloading and saving pre-trained embeddings
    filename = url.split("/")[-1]
    logging.info(f"Beginning download of {filename}")
    # with urllib.request.urlopen(url) as response, open('glove.zip', 'wb') as out_file:
    #     shutil.copyfileobj(response, out_file)
    with requests.get(url, stream=True) as r:
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        block_size = 4096
        progress_bar = tqdm(desc=url.split('/')[-1], miniters=1, total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open("glove.zip", "wb") as f:
            for data in r.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
        
    logging.info("Extracting zip file")
    
    with zipfile.ZipFile("glove.zip", "r") as zip_ref:
        zip_ref.extractall("/mnt")


if __name__ == "__main__":
    run_pipeline()