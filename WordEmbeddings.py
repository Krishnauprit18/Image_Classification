import tensorflow as tf
from tensorflow import keras
from keras import layers

# Using Embedding Layer:-
# This lsyer takes atleat 2 arg.
# the num. of possible words in the vocab., here 1000(1+max. word index),
# and the dimensionality of the embedding, here 32
embedding_layer = layers.Embedding(1000, 32)

# we will train a sentiment classifier on IMDB movie reivews.
vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = vocab_size)
print(train_data[0])
print()

# converting integers back into the text:-
# A dictionary mapping words to an integer index:
word_index = imdb.get_word_index()
# the first indices are reversed:
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK'] = 2  # unknown
word_index['<UNUSED>'] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reversed_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))
print()

# movie reviews can be of diff. lengths so we will use pad_seq. funct. to standerdize the lengths of the reviews.
max_len = 500

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                         value = word_index['<PAD>'],
                                                         padding = 'post',
                                                         maxlen = max_len)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                         value = word_index['<PAD>'],
                                                         padding = 'post',
                                                         maxlen = max_len)

print(train_data[0])
print()

# creating a simple model:
embedding_dim = 16

model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length = max_len),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])

model.summary()

# Compile and train the model:-
model.compile(optimizer = 'adam',
              loass = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(train_data,
                    train_labels,
                    epochs = 30,
                    batch_size = 512,
                    validation_data = (test_data,test_labels))

# Retrieve the learned embeddings:-
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)    # shape: (vocab_size, embedding_dim)
