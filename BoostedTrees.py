# Predicting prob. of passenger survival in titanic dataset:-
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
tf.random.set_seed(123)         # it ensures that across multiple runs we will get the same results.

# Creating feature columns and input functions:-
fc = tf.feature_column
categorical_cols = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numeric_cols = ['age','fare']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab)
    )
feature_columns = []
for feature_name in categorical_cols:
    # we need to one-hot encode the categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in numeric_cols:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))

# viweing the transformations that a feature column produces:
example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class',('First','Second','Third')))
print('Feature val.: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
print()

# dense feature representation of the feature columns:-
print(tf.keras.layers.DenseFeatures(feature_columns)(example).numpy())

# Creating an input function:-
# use entire batch since this is much small dataset:-
numerical_examples = len(y_train)

def make_input_fn(x, y, n_epochs = None, shuffle = True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        if shuffle:
            dataset = dataset.shuffle(numerical_examples)
        # for training cycle through dataset as many times as need (n_epochs = None).
        dataset = dataset.repeat(n_epochs)
        # in memory training doesn't use batching.
        dataset = dataset.batch(numerical_examples)
        return dataset
    return input_fn

# Training and evaluating input functions:-
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle = False, n_epochs = 1)

# Training and evaluating model:- We will build logistic regression classifier to establish a baseline for this problem.
linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps = 100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
#clear_output()
print(pd.Series(result))
print()

# Creating Boosted trees model:- Using BoostedTreesClassifier
# since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer = n_batches)        # type: ignore

# the model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps = 100)

# Eval.
result = est.evaluate(eval_input_fn)
print(pd.Series(result))