import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

keras = tf.keras
tfds.disable_progress_bar()

split_weights = (8,1,1) # 80% train, 10% val., 10% test data.
train_split = tfds.core.utils.subsplit(tfds.Split.TRAIN, weighted=split_weights)         

(raw_train, raw_val, raw_test), metadata = tfds.load('cats_vs_dogs', split = list(splits), with_info = True, as_supervised = True)  # type: ignore
print(raw_train)
print(raw_val)
print(raw_test)

get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    
# Format data:-
img_size = 160  # All images are resized to 160X160 pixel.

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) -1
    image = tf.image.resize(image, (img_size, img_size))
    return image, label

train = raw_train.map(format_example)
validation = raw_val.map(format_example)
test = raw_test.map(format_example)

batch_size = 32
shuffle_buffer_size = 1000

train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
val_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

for image_batch, label_batch in train_batches.take(1):
    pass
image_batch.shape


# Building Model:-
image_shape = (img_size, img_size, 3)

# create the base model from the pre-trained model MobileNet V2.
base_model = tf.keras.applications.MobileNetV2(input_shape = image_shape,
                                               include_top = False,
                                               weights = 'imagenet')

base_model.summary()

# Feature Extractor:
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# Freeze the convulational Base:-
base_model.trainable = False
# base model architecture
base_model.summary()


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Adding prediction layer:
prediction_layer = keras.layers.Dense()
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# stack the feature extractor and add 2 layers:
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Compiling the model:-
base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.summary()

# training the model:-
num_train, num_val, num_test = (
    metadata.splits['train'].num_examples*weight/10
    for weight in split_weights
)
initial_epochs = 10
steps_per_epoch = round(num_train)//batch_size
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
print('initial loss: {:.2f}'.format(loss0))
print('initial accuracy: {:.2f}'.format(accuracy0))
history = model.fit(train_batches,
                    epochs = initial_epochs,
                    validation_data = validation_batches)

# Un-freeze the top layers of the model:-
base_model.trainable = True

# Looking the num. of layers in the base model:-
print('Num. of layers in the base model: ', len(base_model.layers))

# fine tuning this layer onwards:-
fine_tune_at = 100

# freezing all the layers before the 'fine_tune_at' layer:-
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate/10),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.summary()


# Training the model for 10 epochs:-
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_epochs = model.fit(train_batches,
                           epochs = total_epochs,
                           initial_epoch = initial_epochs,
                           validation_data = validation_batches)
