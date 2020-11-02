import click
import urllib.request
import shutil
import zipfile
import os


@click.command()
@click.option('--url', default="http://nlp.stanford.edu/data/glove.42B.300d.zip")
@click.option('--embedding-weights', default="glove.42B.300d.txt")
def run_pipeline(
        url,
        embedding_dimensions_file):

    # Downloading and saving pre-trained embeddings
    with urllib.request.urlopen(url) as response, open('glove.zip', 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    if not os.path('/mnt/glove').exists():
        os.makedirs('/mnt/glove')
    
    with zipfile.ZipFile('glove.zip', 'r') as zip_ref:
        zip_ref.extractall('/mnt/glove')
    

if __name__ == "__main__":
    run_pipeline()