import tensorflow as tf
import datetime,os
from tensorflow import keras
from keras import datasets,layers, models


# Downloading and importing the MNIST dataset: We will reshape each image into a 3D tensor so that input becomes a 4D tensor.We are dealing with grey scale images
# of depth is 1 and width and height are both 28.

(train_img, train_labels), (test_img,test_labels) = datasets.mnist.load_data() 

print('Before Reshaping:')
print('======================')
print('No. of axis on train_img:', train_img)
print('No. of axes on test_img:', test_img)
print('Shape of train_img:', train_img.shape)
print('Shape of test_img:', test_img.shape)
print()

train_img = train_img.reshape((60000,28,28,1))
test_img = test_img.reshape((10000,28,28,1))

print('After Reshaping:')
print('=====================')
print('No. of axis on train_img:', train_img.ndim)
print('No. of axes on test_img:', test_img.ndim)
print('Shape of train_img:', train_img.shape)
print('Shape of test_img:', test_img.shape)
print()

# Creating Convulational Base:-
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()

# Training of model:-
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_img,train_labels, epochs = 5)

# Evaluating the model:-
test_loss, test_acc = model.evaluate(test_img,test_labels)
print(test_acc)
print()
print(test_loss)

# Comparison with Feed Forward NN:-
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train / 255.0, x_test / 255.0

model_dnn = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model_dnn.compile(optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])

model_dnn.summary()
