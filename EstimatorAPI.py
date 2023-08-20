# iris flowers classification model:-

import tensorflow as tf
import pandas as pd
import numpy as np

csv_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
species = ['Setosa', 'Versicolor', 'Verginica']

# downloading and parsing data:-
train_path = tf.keras.utils.get_file(
    'iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = tf.keras.utils.get_file(
    'iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

train = pd.read_csv(train_path, names = csv_column_names, header = 0)
test = pd.read_csv(test_path, names = csv_column_names, header = 0)

print(train.head())
print()

# splitting out the labels for each datasets.
train_y = train.pop('Species')
test_y = test.pop('Species')

# the label column has now been removed from its features:
print(train.head())
print()

# Using Estimator:-
# Creating imput func.:
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth': np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth': np.array([2.2, 1.0])}
    labels = np.array([2,1])
    return features, labels

# Uisng pandas to create input func:
def input_fn(features, labels, training = True, batch_size = 256):          # type: ignore
    "An input func. for training and evaluating"
    # Convert the inputs to a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # Shuffle and repeat if we are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat
        
    return dataset.batch(batch_size)

# Defining Feature Cols.:-
# feature cols. describe how to use input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key = key))

# Instantiate an Estimator:-
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNEstimator(
    feature_columns = my_feature_columns,
    # 2 hidden layers of 10 nodes each.
    hidden_units = [30, 30],
    # the model must choose between 3 classes.
    n_classes = 3        # type: ignore
)                       # type: ignore

# training the model:-
classifier.train(
    input_fn = lambda: input_fn(train, train_y, training = True),
    steps = 5000
)

# Evaluating the model:-
eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(test, test_y, training = False)
)

print('test set accuracy: {accuracy:0.3f}'.format(**eval_result))           # type: ignore

# Making Predictions:-
expected = ['Setosa', 'Versicolor', 'Virginica']

predict_x = {   'SepalLength': [5.1, 5.9, 6.9],
                'SepalWidth': [3.0, 3.0, 3.1],
                'PetalLength': [1.7, 4.2, 5.4],
                'PetalWidth': [0.5, 1.5, 2.1]
            }

def input_fn(features, labels, training = True, batch_size = 256):          # type: ignore
    "An input func. for training and evaluating"
    # Convert the inputs to a dataset
    dataset = tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn = lambda: input_fn(predict_x)
)

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        species[class_id], 100*probability, expec))
