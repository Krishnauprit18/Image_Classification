# We will classify movie reviews into positive and negative using movie review dataset.
# We will use transfer learning: Its is a type of learning where we use some already trained model as a black box to train a new model.
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print('varsion: ', tf.__version__)
print('Eager mode: ',tf.executing_eagerly())
print('Hub version: ', hub.__version__)
print('GPU is', 'available' if tf.test.is_gpu_available() else 'NOT AVAILABLE')

# Downloading IMDB dataset and splitting them into 60% and 40% i.e. 15000
# examples for trainng, 10000 examples for validation and 25000 for testing.
train_val_split = tfds.Split.TRAIN.subsplit([6,4])                                      # type: ignore

(train_data, val_data), test_data = tfds.load(
    name = 'imdb_reviews',
    split = (train_val_split, tfds.Split.TEST),
    as_supervised = True)                                                               # type: ignore

# Exploring data:-
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)

# Build Model:-
embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(embedding, input_shape = [],
                           dtype = tf.string, trainable = True)
hub_layer(train_examples_batch[:3])

# building full mode:-
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

print(model.summary())

# compiling model:-
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# training model:-
history = model.fit(train_data.shuffle(1000).batch(512),
                    epochs = 20,
                    validation_data = val_data.batch(512),
                    verbose = 1)

# Evaluating the model:-
results = model.evaluate(test_data.batch(512), verbose = 0)                 # type: ignore
for name,val in zip(model.metrics_names, results):                          # type: ignore
    print('%s: %.3f' %(name, val))
