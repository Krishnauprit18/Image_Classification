# Doing Feature Extraction and Fine Tuning by using tf.hub:-
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from keras import layers
import numpy as np
import PIL.Image as Image

# Downloading the classifier:-
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/classification/5"
image_shape = (224,224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape = image_shape+(3,))
])

grace_hopper = tf.keras.utils.get_file('image.jpg',"https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
grace_hopper = Image.open(grace_hopper).resize(image_shape)
print(grace_hopper)

grace_hopper = np.array(grace_hopper) / 255.0
grace_hopper.shape
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape

predicted_class = np.argmax(result[0], axis = -1)
print(predicted_class)

# Decoding the predictions:-
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt","https://storage.googleapis.com/download.tensorflow,org/data/ImageNetLabels.txt")
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title('Prediction: '+ predicted_class_name.title())
