# Overfitting happens when our model has excess capacity to memorize the entire trainng data.
# both trianing and val. eorror starts but at a point trianing error goeas down but val. error goes up.
# Underfitting happens when our models doesn't have enough capacity to learn the training data.
# Both training and val. eorror are high.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Downloading IMDB dataset:-
# we will use multi hot encoding which  turns 10000 words into vectora of 0's and 1's.
numb_words = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words = numb_words)

def multi_hot_sequences(sequences, dimension):
    # create an all 0 metrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0              # set specific indices of results[i] to 1's
    return results

train_data = multi_hot_sequences(train_data, dimension = numb_words)
test_data = multi_hot_sequences(test_data, dimension = numb_words)

# looking multi hot vectors:
plt.plot(train_data[0])

# building models:-
baseline_model = keras.Sequential([
    # input shape is only req. here so that '.summary works
    keras.layers.Dense(16, activation = 'relu', input_shape = (numb_words,)),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])

baseline_model.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy', 'binary_crossentropy'])

baseline_model.summary()

# trianng model:-
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs = 20,
                                      batch_size = 512,
                                      validation_data = (test_data, test_labels),
                                      verbose = 2)                                  # type: ignore

# this is kind of overfitting as our loss(training loss) is going down but val. loss is still going up.

# one way to stop overfitting is to reducing the num. of params. in the model.
# other way is to get some kind of training data.
# if training data is not available then we can use regularization techniques (l1, l2, dropout).
# in l1 regularization we add penalty that is prop. to the absolute val. of the param.(or sum of abs. val. of the pramas.)
# l2 regu. we add penalty that is prop. to sum of square of the val. of the params.
# in dropout regu. we decide to randomly drop certain nodes from the hiddden layer or input layer of NN. we normally set dropout 
# between 20% to 50% (means 20% to 50% of nodes will be randomly dropped in each iteration in the NN training.)

# creating a smaller model:- with less hidden units to comapre with the baseline model.
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation = 'relu', input_shape = (numb_words,)),
    keras.layers.Dense(4, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

smaller_model.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['accuracy','binary_crossentropy'])

smaller_model.summary()

# training this smaller model:-
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs = 20,
                                    batch_size = 512,
                                    validation_data = (test_data, test_labels),
                                    verbose = 2)                        # type: ignore

# still model is overfitted so we obs. that at 5th epoch model starts to be overfitted.

# creating bigger model:-
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation = 'relu', input_shape = (numb_words,)),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

bigger_model.compile(optimizer = 'adam',
                     loss = 'binary_crossentropy',
                     metrics = ['accuracy','binary_crossentropy']
                    )
                    
bigger_model.summary()

# training the model:-
bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs = 20,
                                  batch_size = 512,
                                  validation_data = (test_data,test_labels),
                                  verbose = 2)                                  # type: ignore


# plotting training and validation loss:-
def plot_history(histories, key = 'binary_crossentropy'):
    plt.figure(figsize = (16,10))
    
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       "--", label = name.title()+'val')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(),
                 label = name.title()+'Train')
        
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    
    plt.xlim([0, max(history.epoch)])

plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

# Adding Regularization:- To stop overfitting.
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001),
                       activation = 'relu', input_shape = (numb_words,)),
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001),
                       activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

l2_model.compile(optimizer = 'adam',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy','binary_crossentropy']
                )

l2_model_history = l2_model.fit(train_data,
                                train_labels,
                                epochs = 20,
                                batch_size = 512,
                                validation_data = (test_data,test_labels),
                                verbose = 2)                                # type: ignore

plot_history([('baseline', baseline_history),
              'l2', l2_model_history])

# We observed that the regularized model is resistant to overfitting for some more epochs
# then the unregularized model.

# Adding Dropout:- It is not applied at the testing model time.
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation = 'relu', input_shape = (numb_words,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = 'sigmoid')
])

dpt_model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data,train_labels,
                                  epochs = 20,
                                  batch_size = 512,
                                  validation_data = (test_data,test_labels),
                                  verbose = 2)      # type: ignore

plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])