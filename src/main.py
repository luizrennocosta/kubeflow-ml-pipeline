#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import nltk
import string
import numpy as np
nltk.download('stopwords')
data_path = 'data/'
import re
import urllib
import requests
import zipfile
#%%
"""
    Column Descriptions:

    * marketplace       - 2 letter country code of the marketplace where the review was written.
    * customer_id       - Random identifier that can be used to aggregate reviews written by a single author.
    * review_id         - The unique ID of the review.
    * product_id        - The unique Product ID the review pertains to. In the multilingual  dataset the reviews
    *                     for the same product in different countries can be grouped by the same product_id.
    * product_parent    - Random identifier that can be used to aggregate reviews for the same product.
    * product_title     - Title of the product.
    * product_category  - Broad product category thatsudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn* can be used to group reviews 
    *                     (also used to group the dataset into coherent parts).
    * star_rating       - The 1-5 star rating of the review.
    * helpful_votes     - Number of helpful votes.
    * total_votes       - Number of total votes the review received.
    * vine              - Review was written as part of the Vine program.
    * verified_purchase - The review is on a verified purchase.
    * review_headline   - The title of the review.
    * review_body       - The review text.
    * review_date       - The date the review was written.
"""
column_names = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date']

url = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_UK_v1_00.tsv.gz"
dataset = pd.read_csv(url, sep='\t', error_bad_lines=False, names=column_names, compression='gzip')

dataset = dataset[['product_title', 'product_category', 'review_body', 'review_headline']]

# dataset.product_category.value_counts().nlargest(10).plot.bar()

# train_data, test_data = train_test_split(dataset, test_size=.10)
# tokenizer = Tokenizer(num_words= 10000, oov_token='<OOV>')
dataset_ = dataset.copy()

#%%
%%time


stemmer = nltk.stem.SnowballStemmer("english")
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

dataset = dataset_
dataset['review_body'] = dataset['review_body'].astype(str)
dataset['review_body'] = dataset['review_body'].str.lower()
dataset['review_body'] = dataset['review_body'].str.replace('[!"#$%&()*\'+,-./:;<=>?@[\\]^_{|}~`\t\n0123456789]', ' ')
dataset['review_body'] = dataset['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))



#%%
embedding_weights = "http://nlp.stanford.edu/data/glove.42B.300d.zip"

with requests.get(embedding_weights, stream=True) as r:
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    block_size = 4096
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open('glove.zip', 'wb') as f:
        for data in r.iter_content(block_size):
            f.write(data)
            progress_bar.update(len(data))
#%%


with zipfile.ZipFile('glove.zip', 'r') as zip_ref:
    zip_ref.extractall('./glove')
#%%

path_to_glove_file = 'glove/glove.42B.300d.txt'
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

#%%
%%time
vocab_size=20000
max_length = 300
class_amount = dataset['product_category'].unique().size

#%%
train_text = dataset.review_body.to_numpy()
tokenizer = Tokenizer(num_words= vocab_size, lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)
#%%

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_text)
train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=max_length)

#%%
num_tokens = len(tokenizer.word_index) + 2
embedding_dim = 300
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

#%%
# * encoding labels
encoder = LabelEncoder()
Y = encoder.fit_transform(dataset['product_category'])
Y = to_categorical(Y)

#%%

seed = np.random.randint(500)

X_train, X_test, Y_train, Y_test= train_test_split(train_padded.astype(int), Y.astype(int), test_size=.10, random_state=seed)
X_train.shape
#%%
from tensorflow.keras import layers
from tensorflow.keras import initializers, Input, Model

embedding_layer = layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=initializers.Constant(embedding_matrix),
    trainable=True,
)

int_sequences_input = Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(class_amount, activation="softmax")(x)
model = Model(int_sequences_input, preds)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=20, batch_size=128)


# %%
