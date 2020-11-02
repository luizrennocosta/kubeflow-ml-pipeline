#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
    * product_category  - Broad product category that can be used to group reviews 
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


dataset = pd.read_csv('../data/amazon_reviews_multilingual_UK_v1_00.tsv', sep='\t', error_bad_lines=False, names=column_names)

dataset = dataset[['product_title', 'product_category', 'review_body', 'review_headline']]

# dataset.product_category.value_counts().nlargest(10).plot.bar()

train_data, test_data = train_test_split(dataset, test_size=.10)
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
%%time
vocab_size=20000
embedding_dim = 32
max_length = 50
class_amount = dataset['product_category'].unique().size

#%%
train_text = dataset.review_body.to_numpy()
tokenizer = Tokenizer(num_words= 20000, lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)

# %%

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_text)
train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=max_length)



#%%
# * encoding labels
encoder = LabelEncoder()
Y = encoder.fit_transform(dataset['product_category'])
Y = to_categorical(Y)

#%%
from sklearn.preprocessing import LabelEncoder

seed = np.random.randint(500)

X_train, X_test, Y_train, Y_test= train_test_split(train_padded.astype(int), Y.astype(int), test_size=.10, random_state=seed)

#%%


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),  # tf.keras.layers.GlobalAveragePooling1D()     quicker than flattern, but less accuracy than falltern
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(class_amount, activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=10, batch_size=5000)


# %%
