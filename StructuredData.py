# Model to predict whether the patient will suffer from heart disease.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras
import sklearn 
from sklearn.model_selection import train_test_split

# using pandas to create a dataframe:-
URL = 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
dataframe = pd.read_csv(URL)
print(dataframe.head())
print()

# splitting data into train, test, val. dataset:-
train, test = train_test_split(dataframe, test_size = 0.2)
train, val = train_test_split(train, test_size = 0.2)
print(len(train), 'train examples')
print()
print(len(val), 'validation examples')
print()
print(len(test), 'test examples')
print()

# creating input data pipeline using tf.data:-
# A utility method to create a tf.data dataset from a pandas dataframe->

def df_to_dataset(dataframe, shuffle = True, batch_size = 32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices(dict(dataframe), labels)
    if shuffle:
        ds = ds.shuffle(buffer_size = len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# converting dataframe to dataset:-
batch_size = 5  # A small batch size is used for demonstration purposes.
train_ds = df_to_dataset(train, batch_size = batch_size)
val_ds = df_to_dataset(val, shuffle = False, batch_size = batch_size)
test_ds = df_to_dataset(test, shuffle = False, batch_size = batch_size)

# Understanding input pipeline:-
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature: ', list(feature_batch.keys()))
    print('A batch of ages: ', feature_batch['age'])
    print('A batch of targets: ',label_batch)
    
    
# Several types of feature cols.:-
# we will use this batch to demonstrate several types of feature cols.:
example_batch = next(iter(train_ds))[0]             # type: ignore

# A utility method to create  a feature col. and to transform a batch for data.
def demo(feature_column):
    feature_layer = keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())
    
# Numeric cols.:
age = feature_column.numeric_column('age')
demo(age)

# Bucketized cols.:
age_buckets = feature_column.bucketized_column(age, boundaries = [18,25,30,35,40,45,50,55,60,65])
demo(age_buckets)

# Categorical cols.:
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed','normal','reversible']
)

that_one_hot = feature_column.indicator_column(thal)
demo(that_one_hot)

# Embedding cols.:
# input to embedding col is categorical col.
that_embedding = feature_column.embedding_column(thal, dimension = 8)
demo(that_embedding)

# Hashed Feature Cols.:
thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size = 1000)
demo(feature_column.indicator_column(thal_hashed))

# Note= indicator_column it is used to create one hot encoding.
# Crossed features cols.(feature crossing):
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size = 1000)
demo(feature_column.indicator_column(crossed_feature))

# Choosing cols. to train our model:
feature_columns = []

# numeric columns:-
for header in ['age','trestbps','chol','thalach','oldpeak','slope','ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized columns:-
age_buckets = feature_column.bucketized_column(age, boundaries = [18,25,30,35,40,45,50,55,60,65])
feature_columns.append(age_buckets)

# indicator columns:-
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal',['fixed','normal','reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding columns:-
that_embedding = feature_column.embedding_column(thal, dimension = 8)
feature_columns.append(that_embedding)

# crosssed columns:-
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size = 1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


# creating feature layer:-
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size = batch_size)
val_ds = df_to_dataset(val, shuffle = False, batch_size = batch_size)
test_ds = df_to_dataset(test, shuffle = False, batch_size = batch_size)

# creating, compiling and training the model:
# creating a baseline model. with logistic regression:-
model = tf.keras.Sequential([
    feature_layer,
    keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              run_eagerly = True)

model.fit(train_ds,
          validation_data = val_ds,
          epochs = 5)

loss, accuracy = model.evaluate(test_ds)
print('Accuracy', accuracy)

# Building NN based model:-
model_NN = tf.keras.Sequential([
    feature_layer,
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

model_NN.compile(optimizer = 'adam',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'],
                 run_eagerly = True)

model_NN.fit(train_ds,
             validation_data = val_ds,
             epochs = 5)

print(model_NN.summary())

loss, accuracy = model_NN.evaluate(test_ds)
print('Accuracy', accuracy)