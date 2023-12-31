# -*- coding: utf-8 -*-
"""5G-IDS-FINAL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15ydvI8Rb_HfjrXvp4AxnV-JPxWVkFsww
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Tensorflow:- It is a framework that supports high and low level API.
# train_test_split:- It is used to split dta into trianing and testing sets. It splits the data
# After randomly shuffling it.It ensures the uniform distribution of data.
# LabelEncoder:- It helps in encoding categorical labels(text fotmat) into numeric values.
# It assigns an integer to each catagory in data. example type of attack is benign and malicous.
# values assigned to these categories is benign-1, malicious-0.
# Min Max Scalar:- It transforms the data by scaling each feature to a range
# typically btw 0 and 1.
# Standard scalar:- It transform the data by scaling the features to have mean of 0
# and standard deviation of 1.

# Loading the dataset:-
ds = pd.read_csv('/content/sample_data/5gnidd.csv', low_memory=False)
ds.head()

ds.tail()

ds.describe()

# Dropping unnecessary columns:-
unnecessary_col = ['Unnamed: 0', 'RunTime', 'Min', 'Max', 'sTos', 'dTos', 'sDSb', 'dDSb', 'sHops', 'dHops', 'SrcWin',
                  'DstWin', 'sVid', 'dVid', 'SrcTCPBase', 'DstTCPBase', 'TcpRtt', 'SynAck', 'AckDat'
                  ]
ds = ds.drop(unnecessary_col, axis = 1)

# Dropping any missing values:-
ds = ds.dropna()

# Checking for any duplicate values:-
print(ds.duplicated().sum())
# dropping any duplicate values:-
ds = ds.drop_duplicates()
print(ds.shape)
print(ds.head())

# Duplicate function():- checks for redundent rows and returns boolean series either Ture or Flase.
# Drop_dulicates:- It drops the duplicate rows.
# Sum():- it counts the Duplicate rows present and gives total counts.

# Separating feature columns (X) and label columns (Y):-
X = ds.drop(['Label','Attack Type','Attack Tool'], axis = 1)
Y = ds['Label']

# Converting any categorical columns into numerical ones using one-hot encoding:-
cat_col = ['Proto','Cause','State']
X = pd.get_dummies(X, columns = cat_col)

# pd.get_dummies():- convert categorical variables into dummy variables.
# X is the feature columns and cat_col is the provided columns to be one-hot encoded.
# It assignes the binary coluns and asigns each columns with values either 0 or 1.
# These values indicates the presence or absence of the features in the columns.
# for example in protocol columns it has 3 categories:- icmp, ud, tcp,
# so the function will create 3 binary columns and assign each catagory with either
# value 1 for presence of that catagory and 0 for absence of that catagory.

# Feature Scaling:-
columns_to_scale = ['Seq','Dur','Mean','Sum','TotPkts','SrcPkts','DstPkts',
                    'TotBytes','SrcBytes','DstBytes','Offset','sMeanPktSz',
                    'dMeanPktSz','Load','SrcLoad','DstLoad','Loss','SrcLoss',
                    'DstLoss','pLoss','Rate','SrcRate','DstRate','SrcGap','DstGap',
                    'sTtl','dTtl'
                    ]
numeric_data = ds[columns_to_scale]
# Min Max Scaling:-
min_max_scaling = MinMaxScaler()
# Applying Z-Score Normalization:-
standard_scaling = StandardScaler()
standard_scaled_data = standard_scaling.fit_transform(numeric_data)
standard_scaled_data = pd.DataFrame(standard_scaled_data, columns = columns_to_scale)

# min max scaling is a feature extracting technique.
# It scales the data into specified range, usually btw. 0 and 1.
# Standard scaler is also a feature scaling technique.
# It standarizes the data by transforming it to have a mean of 0 and a standard
# deviation of 1.
# Z-Score normalization:-
#calculates the mean and standard deviation of each column in 'numeric_data'
# and then applies the standardization formula to each value.

# Converting dataframe to numpy array:-
X = X.to_numpy()

# Converting labels to numerical values:-
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Label Encoder first fits the encoder to the data  to learn the mapping
# between unique labels and numerical values, and then it transforms the labels
# to their corresponding numerical representation.
# During the fitting process, the labelencoder identifies all unique labels in
# and assigns a unique integer to each label.
# The transformation process replaces each label in Y with its corresponding integer
# representation.

# Converting target labels to one-hot encoding;-
numerical_classes = len(label_encoder.classes_)
Y = tf.keras.utils.to_categorical(Y, numerical_classes)

# calculates the number of unique classes in the target variable Y
# after it has been encoded.
# The label_encoder.classes_ attribute contains the unique classes that were encountered
# during the label encoding process.
# len(label_encoder.classes_) gives us the total count of unique classes.
# This line applies one-hot encoding (one-of-K encoding) to the numerical
# encoded target variable Y.
# The to_categorical() function is used for this purpose.

# This function converts a class vector Y to a one-hot encoded matrix.
# In the one-hot encoded matrix, each row corresponds to a sample, and each column corresponds to a unique class from the target variable.
# The value in the matrix is 1 if the corresponding sample belongs to that class,
# or 0 otherwise.

# Splitting dataset into training and testing part:-
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.3,
    random_state = 42
)

# Random state is used to seed the random number generator.
# Providing a specific value for random state ensures that the data split remains consistent
# each time the code is run.
# This helps in reproducibility, as the same random data points will be assigned to the training and testing sets
# whenever the code is executed with the same random_state value.

# Reshaping input data for our RNN(assuming that the dataset hai 1-D sequences):-
input_shape = (X_train.shape[1],1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# is a tuple that defines the shape of the input data for the RNN.
# The first element of the tuple, X_train.shape[1],
# represents the number of time steps in each sequence, and the second element, 1, represents the number of input features
# at each time step. In this case, it indicates that each sequence in the dataset is 1-dimensional.

# X_train.shape[0]:- The first dimension of 'X_train' represents the number of samples in the training set (i.e., the batch size). X_train.shape[0] gives us the number of samples.

# X_train.shape[1]:- The second dimension of 'X_train' represents the number of time steps in each sequence. In this step, we want to transform each sample from a 2-dimensional sequence to a 3-dimensional sequence with one feature at each time step.

# X_train.reshape(X_train.shape[0], X_train.shape[1], 1):- This function call reshapes 'X_train' into a 3-dimensional array with shape (batch_size, time_steps, input_features). Here, the number of input features is set to 1, as we previously defined in the 'input_shape' tuple.

# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1):
# Similarly, this line reshapes the testing data 'X_test' to have the same 3-dimensional format as the training data. It ensures that the RNN can process the testing data in the same sequence format as the training data.

# Building our RNN Model:-
rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape = input_shape),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(numerical_classes, activation = 'softmax')
])

# Model compilation:-
rnn_model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['Accuracy']
                  )
'''
tf.keras.layers.LSTM(64, input_shape=input_shape):-
This is the Long Short-Term Memory (LSTM) layer, a type of RNN cell that is particularly effective at capturing long-term dependencies in sequential data. The LSTM layer has 64 units (or cells), which determines the dimensionality of the hidden state.

64:- This is the number of LSTM units in the layer. It represents the dimensionality of the LSTM's internal representation, also known as the hidden state. More units allow the model to learn more complex patterns, but they also increase the number of parameters and training time.

input_shape: This specifies the shape of the input data that will be fed into the model. It should be in the format (time_steps, input_features). In this case, the input_shape is derived from the variable defined earlier, representing the number of time steps and input features in the data.

b. tf.keras.layers.Dense(128, activation='relu'):
This is a fully connected (dense) layer with 128 units and the ReLU activation function. Dense layers are used for learning high-level abstractions from the output of the previous LSTM layer. The ReLU activation function introduces non-linearity, which allows the model to learn complex relationships in the data.

c. tf.keras.layers.Dense(numerical_classes, activation='softmax'):
This is the output layer of the model with numerical_classes units and the softmax activation function. The number of units in this layer corresponds to the number of classes in the target variable ('Y'). The softmax activation function produces a probability distribution over the classes, allowing the model to make multiclass predictions.

Model Compilation:
After building the model, the next step is to compile it. During compilation, we specify the loss function, the optimizer, and the metrics to be used during training.

a. loss='categorical_crossentropy':
This is the loss function used for categorical multiclass classification problems. Since the target variable 'Y' has been one-hot encoded, categorical cross-entropy is an appropriate choice for the loss function.

b. optimizer='adam':
The Adam optimizer is chosen as the optimization algorithm. Adam is a popular and effective optimization algorithm that adapts the learning rate during training.

c. metrics=['Accuracy']:
The metric used to monitor the model's performance during training is accuracy. Accuracy measures the proportion of correct predictions made by the model over the total number of predictions.
'''

# Training rnn model:-
epochs = 150
batch_size = 50
history = rnn_model.fit(X_train,
                        Y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_split = 0.3
                        )

# Model evaluation:-
loss, accuracy = rnn_model.evaluate(X_test, Y_test, batch_size = batch_size)
print(f'Test loss is: {loss:.4f}, Test accuracy is: {accuracy:.4f}')

# making predictions:-
y_pred = rnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_test_classes = np.argmax(Y_test, axis = 1)
print(y_pred_classes)
print()
print(y_test_classes)

'''
y_pred = rnn_model.predict(X_test):
This line uses the trained RNN model (rnn_model) to make predictions on the test data (X_test). The predict() method of the model is used to obtain the predicted probabilities for each class for each sample in the test data. The variable y_pred will now hold a 2-dimensional array, where each row represents a sample from the test data, and each column contains the predicted probabilities for each class.

y_pred_classes = np.argmax(y_pred, axis=1):
The np.argmax() function is used to convert the predicted probabilities (y_pred) into class labels (y_pred_classes). It performs this conversion by finding the index of the maximum probability along each row (axis 1) of the y_pred array. This index corresponds to the class with the highest predicted probability for each sample. So, y_pred_classes will be a 1-dimensional array containing the predicted class labels for each sample in the test data.

y_test_classes = np.argmax(Y_test, axis=1):
Similar to the previous step, this line uses the np.argmax() function to convert the one-hot encoded ground truth labels (Y_test) into their corresponding class labels (y_test_classes). The axis=1 parameter specifies that the maximum value is to be found along the second axis (columns) of the Y_test array. This step is necessary because the original labels were one-hot encoded, and we need to convert them back to their numerical class representation.

print(y_pred_classes):
This line prints the predicted class labels for each sample in the test data. The y_pred_classes array will contain the class labels predicted by the model for each test sample.

print():
This line prints an empty line, creating a separation between the two arrays printed for clarity.

print(y_test_classes):
This line prints the ground truth class labels for each sample in the test data. The y_test_classes array will contain the actual class labels for each test sample.
'''

# Converting prediction labels into original labels:-
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
y_test_labels = label_encoder.inverse_transform(y_test_classes)
print(y_pred_labels)
print()
print(y_test_labels)

'''
y_pred_labels = label_encoder.inverse_transform(y_pred_classes):
The inverse_transform() method of the LabelEncoder object (label_encoder) is used to convert the numerical class labels (y_pred_classes) back into their original categorical labels (y_pred_labels). This method reverses the encoding process performed earlier when the labels were converted into numerical representations.
label_encoder: This is the LabelEncoder object used earlier to encode the categorical labels into numerical values.

y_pred_classes: This is the array containing the predicted class labels (numerical representations) obtained from the RNN model's predictions on the test data.

The inverse_transform() method maps the numerical labels in y_pred_classes back to their original categorical labels, producing an array (y_pred_labels) with the same shape as y_pred_classes but with the categorical class labels for each sample.

y_test_labels = label_encoder.inverse_transform(y_test_classes):
Similar to the previous step, this line uses the inverse_transform() method of the LabelEncoder object (label_encoder) to convert the numerical ground truth class labels (y_test_classes) back into their original categorical labels (y_test_labels).
label_encoder: The same LabelEncoder object used throughout the code.

y_test_classes: This is the array containing the ground truth class labels (numerical representations) derived from the original one-hot encoded ground truth labels (Y_test).

The inverse_transform() method maps the numerical labels in y_test_classes back to their original categorical labels, producing an array (y_test_labels) with the same shape as y_test_classes but with the categorical class labels for each sample.

print(y_pred_labels):
This line prints the predicted categorical labels (y_pred_labels) for each sample in the test data. The y_pred_labels array will contain the predicted class labels as strings, representing the RNN model's predicted class for each test sample.

print():
This line prints an empty line, creating a separation between the two arrays printed for clarity.

print(y_test_labels):
This line prints the ground truth categorical labels (y_test_labels) for each sample in the test data. The y_test_labels array will contain the actual class labels for each test sample, representing the true labels of the data.
'''

# Calculating accuracy:-
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print(f'Accuracy is: {accuracy:.4f}')

# Classification report and confusion matrix:-
classification_rep = classification_report(y_test_labels, y_pred_labels)
print('Classification report is:-')
print()
print(classification_rep)
print()

confusion_mat = confusion_matrix(y_test_labels, y_pred_labels)
print('Confusion matrix is:-')
print()
print(confusion_mat)