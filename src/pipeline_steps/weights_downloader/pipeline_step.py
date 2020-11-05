import click
import zipfile
import logging
import requests
from tqdm import tqdm


@click.command()
@click.option("--url", default="http://nlp.stanford.edu/data/glove.42B.300d.zip")
def run_pipeline(url):

    # Downloading and saving pre-trained embeddings
    filename = url.split("/")[-1]
    logging.info(f"Beginning download of {filename}")
    # with urllib.request.urlopen(url) as response, open('glove.zip', 'wb') as out_file:
    #     shutil.copyfileobj(response, out_file)
    with requests.get(url, stream=True) as r:
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        block_size = 4096
        progress_bar = tqdm(
            desc=url.split("/")[-1],
            miniters=1,
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
        )
        with open("glove.zip", "wb") as f:
            for data in r.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))

    logging.info("Extracting zip file")

    with zipfile.ZipFile("glove.zip", "r") as zip_ref:
        zip_ref.extractall("/mnt")


if __name__ == "__main__":
    run_pipeline()
