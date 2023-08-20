import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# Loading Data:-
_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin = _URL, extract = True)
path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# After extracting contents we assign var. with proper file path for training and val. set:-
train_dir = os.path.join(path, 'train')
val_dir = os.path.join(path, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')    # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')    # directory with our training dog pictures
validation_cats_dir = os.path.join(val_dir, 'cats')     # dir. with our val. cat pics.
validation_dogs_dir = os.path.join(val_dir, 'dogs')     # dir. with our val. dog pics.

# Understanding our data:-
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr+num_dogs_tr
total_val = num_cats_val+num_dogs_val

print('total training cat images: ', num_cats_tr)
print('total training dog images: ', num_dogs_tr)
print()
print('total val. cat images: ', num_cats_val)
print('total val. dog images: ', num_dogs_val)
print()
print('total training images: ', total_train)
print('total val. images: ', total_val)
print()

batch_size = 128
epochs = 15
img_height = 150
img_width = 150

# Data Preparation:-
train_image_gen = ImageDataGenerator(rescale = 1. / 255)    # Generator for our training data
val_image_gen = ImageDataGenerator(rescale = 1. / 255)      # Generator for our val. data

train_data_gen = train_image_gen.flow_from_directory(batch_size = batch_size,
                                                     directory = train_dir,
                                                     shuffle = True,
                                                     target_size = (img_height, img_width),
                                                     class_mode = 'binary')
val_data_gen = val_image_gen.flow_from_directory(batch_size = batch_size,
                                                 directory = val_dir,
                                                 target_size = (img_height, img_width),
                                                 class_mode = 'binary')

# Visualising training images:-
sample_training_images, _ = next(train_data_gen)

# function to plot images in the form of a grid with 1 row and 5 cols. where images are placed in each col.:-
def plot_images(images_arr):
    fig, axes = plt.subplots(1,5, figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):       # type: ignore
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_images(sample_training_images[:5])

# Creating the model:-
model = Sequential([
    Conv2D(16, 3, padding = 'same', activation = 'relu', input_shape = (img_height, img_width, 3)),
    MaxPool2D(),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPool2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPool2D(),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

# Compile the model:-
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.summary()

# Train the model:-
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch = total_train // batch_size,
    epochs = epochs,
    validation_data = val_data_gen,
    validation_steps = total_val // batch_size
)

# visualizing training results:-
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize = (8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, acc, label = 'Training loss')
plt.plot(epochs_range, val_acc, label = 'Validation loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation loss')
plt.show()

# our model is overfitted so we will use data augmentation to resolve overfittng:-

# random horizaontal flip on dataset:
img_gen = ImageDataGenerator(rescale = 1. / 255, horizontal_flip = True)
train_data_gen = img_gen.flow_from_directory(batch_size = batch_size,
                                             directory = train_dir,
                                             shuffle = True,
                                             target_size = (img_height, img_width))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# reuse the same custom plotting func. defined and used
# above to visualize the training images:
plot_images(augmented_images)

# Randomly rotating image(45 deg.):
img_gen = ImageDataGenerator(rescale = 1. / 255, rotation_range = 45)
train_data_gen = img_gen.flow_from_directory(batch_size = batch_size,
                                             directory = train_dir,
                                             shuffle = True,
                                             target_size = (img_height, img_width))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# Applying zoom augmentation:
img_gen = ImageDataGenerator(rescale = 1. / 255, zoom_range = 0.5)
train_data_gen = img_gen.flow_from_directory(batch_size = batch_size,
                                             directory = train_dir,
                                             shuffle = True,
                                             target_size = (img_height, img_width))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# Put it all together:-
img_gen_train = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 45,
    width_shift_range = .15,
    height_shift_range = .15,
    horizontal_flip = True,
    zoom_range = 0.5
)

train_data_gen = img_gen_train.flow_from_directory(batch_size = batch_size,
                                                   directory = train_dir,
                                                   shuffle = True,
                                                   target_size = (img_height, img_width),
                                                   class_mode = 'binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

# Creating validation data gen.:
img_gen_val = ImageDataGenerator(rescale = 1. / 255)
val_data_gen = img_gen_val.flow_from_directory(batch_size = batch_size,
                                               directory = val_dir,
                                               target_size = (img_height, img_width),
                                               class_mode = 'binary')

# Another technique to resolve model overfitting is Dropout:-
model_new = Sequential([
    Conv2D(16, 3, padding = 'same', activation = 'relu',
           input_shape = (img_height, img_width, 3)),
    MaxPool2D(),
    Dropout(0.3),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPool2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPool2D(),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.1),
    Dense(1, activation = 'sigmoid')
])

# compiling the model:-
model_new.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy']
                )

model_new.summary()

history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch = total_train // batch_size,
    epochs = epochs,
    validation_data = val_data_gen,
    validation_steps = total_val // batch_size
)

model_new.layers
layer_outputs = [layer.output for layer in model_new.layers[:8]]
print('layer outputs is: ', layer_outputs)
layer_names = [layer.name for layer in model_new.layers[:8]]
print('layer name is: ', layer_names)
activation_model = tf.keras.models.Model(inputs = model_new.input, outputs = layer_outputs)

activations = activation_model.predict(train_data_gen[0])
plt.imshow(train_data_gen[0][0][0])
plt.show()

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 3], cmap = 'viridis')
plt.show()

plt.matshow(first_layer_activation[0, :, :, 15], cmap = 'viridis')
plt.show()
