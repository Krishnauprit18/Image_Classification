# Regression Problem:-
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import pathlib

# Getting data:-
dataset_path = keras.utils.get_file("auto-mpg.data","https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

# reading data using pandas:-
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = '?', comment = '\t', sep = " ", skipinitialspace = True)
dataset = raw_dataset.copy()
print(dataset.tail())
print()


# Cleaning data:-
# checking null val. col.->
print(dataset.isna().sum())
print()
dataset = dataset.dropna()
print(dataset)
print()

# converting "Origin" into One-Hot encoding:-
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1)*1.0
dataset["Europe"] = (origin == 2)*1.0
dataset["Japan"] = (origin == 3)*1.0
print(dataset.tail())
print()

# splitting datset into traning and testing data:-
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)


# Inspecting data:- Looking at joint distribution of a few pairs of col. from training dataset->
sns.pairplot(train_dataset[["MPG","Cylinders","Displacement","Weight"]], diag_kind = "kde")

# Looking overall statistics:-
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)
print()

# spliting features from labels:-
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalization of Data:-
def norm(x):
    return (x-train_stats['mean'])/ train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Building Model:-
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation = 'relu', input_shape = [len(train_dataset.keys())]),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

# inspecting model:-
model.summary()
print()

# predicting model:-
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print('result is: ', example_result)
print()

# training model for 1000 epochs:-
# Display trainng progress by printing a single dot for each completed epoch:-
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch%100 == 0:
            print('')
        else:
            print('.', end = '')
EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs = EPOCHS, validation_split = 0.2, verbose = 0,       # type: ignore
    callbacks = [PrintDot()]
    )

# visualizing model:-
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
print()

# plotting history:-
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0.5])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'],hist['mse'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()
    
plot_history(history)

# Our model is in overfitting case so we will use EarlyStopping call back(regularization).
model = build_model()

# The patience param. is the amount to check for improvement->
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)

history = model.fit(normed_train_data, train_labels, epochs = EPOCHS,
                    validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])   # type: ignore
plot_history(history)

# Evaluating data:-
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0)     # type: ignore
print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))


# Using model to Predict MPG:-
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100,100],[-100,100])


# error distribution plot:-
error = test_predictions-test_labels
plt.hist(error, bins = 25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


