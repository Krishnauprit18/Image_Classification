# Classification Problem:-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print()

# import fashion mnist dataset for model training:
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_img,train_labels),(test_img,test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# exploring data:-
print(train_img.shape)
print()
print(len(train_labels))
print()
print(train_labels)
print()
print(test_img.shape)
print()
print(len(test_labels))
print()
print(test_labels)
print()

# preprocessing data:-
plt.figure()
plt.imshow(train_img[0])
plt.colorbar()
plt.show()
'''
plt.figure()
plt.imshow(train_img[1])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[2])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[3])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[4])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[5])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[6])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[7])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[8])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(train_img[9])
plt.colorbar()
plt.show()
'''


# scaling these pixel values ranging from 0 to 1. to do so we  will divide the val. by 255.
train_img = train_img/255.0
test_img = test_img/255.0
print('nor,alized train img is: ',train_img)
print()
print('normalized test  img is: ',test_img)


# we will be plotting first 25 img. from traning set. And display class name below eaxh img.:
plt.figure(figsize = (10,10))                               
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap = plt.cm.binary)      # type: ignore
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Building Model:- setting up layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])


# compiling the model:
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


# Training the model:
model.fit(train_img,train_labels,epochs = 10)


# evaluating accuracy:
test_loss, test_acc = model.evaluate(test_img,test_labels)
print('Model test accuracy is: ',test_acc)


# Making predictions:
predictions = model.predict(test_img)
np.argmax(predictions[0])
print(test_labels[0])

# plotting predicred and actual val.:
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img,cmap = plt.cm.binary)            # type: ignore
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
               class_names[true_label]),
               color = color)
    
def plot_val_arr(i,predictions_array, true_label):
    predictions_array, true_label = predictions_array[i],true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize = (6,3))
plt.subplot(1,2,1)
plot_image(i,predictions, test_labels, test_img)
plt.subplot(1,2,2)
plot_val_arr(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize = (6,3))
plt.subplot(1,2,1)
plot_image(i,predictions, test_labels, test_img)
plt.subplot(1,2,2)
plot_val_arr(i, predictions, test_labels)
plt.show()


# plot the first X test images, thi=eir predicted labels, and the true labels.
# color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize = (2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i,predictions, test_labels, test_img)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_val_arr(i, predictions, test_labels)
plt.show()


# making predcitions for single image:
# grab an image from image dataset:
img = test_img[0]
print('image shape is: ',img.shape)

# adding image to a batch where its the only number.
img = (np.expand_dims(img,0))
print('batch image shape is: ',img.shape)

# predicting correct label for this img.:
predictions_single = model.predict(img)
print(predictions_single)
plot_val_arr(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)
np.argmax(predictions_single[0])