import matplotlib.pyplot as plt      
import numpy as np
import pandas as pd
import tensorflow as tf

# Training Workflows:-
# Processing multiple epochs:  while trainig any ml model or NN we make multiple pass over the dataset.
# One complete pass over a dataset is known as epoch.
# creating a dataset that repeats its inputs for 3 epochs.

# Constructing a datset from csv file:
titanic_file = tf.keras.utils.get_file("train.csv","https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

# defining a func.to plot batch size:
def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)),batch_sizes)
    plt.xlabel('Batch Number')
    plt.ylabel('Batch Size')
    
'''
1) The dataset.repeat() transformation in TensorFlow is used to repeat the elements of a dataset for a specified number of epochs or indefinitely if no argument is provided.

2) When you apply the repeat() transformation to a dataset, it creates a new dataset that iterates over the elements of the original dataset repeatedly. Each pass through the dataset is considered as one epoch. The number of epochs is determined by the argument passed to the repeat() function.

3) If you don't provide an argument to repeat(), the dataset will repeat indefinitely. This is useful when you want to repeatedly train a model on the same dataset until a stopping condition is met, such as reaching a certain number of steps or achieving a desired level of performance.
'''

titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)

# Applying repeat after batch:
titanic_batches = titanic_lines.batch(128).repeat(3)
plot_batch_sizes(titanic_batches)

# printing shape of batch and also printing id of epoch at the end of epoch:
epochs = 3
dataset = titanic_lines.batch(128)

for epoch in range(epochs):
    for batch in dataset:
        print(batch.shape)           # type: ignore
    print('End of epoch: ',epoch)
print()
'''
1) Shuffle:The shuffle() transformation in TensorFlow is used to randomly shuffle the elements of a dataset. It is commonly used to introduce randomness and reduce bias in the order of training examples during the training process.

2) When you apply the shuffle() transformation to a dataset, it creates a new dataset that iterates over the elements of the original dataset in a random order. The buffer size passed as an argument to shuffle() determines the number of elements from the dataset that are loaded into a buffer. TensorFlow then randomly samples elements from this buffer to form a shuffled dataset.

3) Note that if the buffer_size is set to a number equal to or larger than the total number of elements in the dataset, the entire dataset will be shuffled completely. On the other hand, if buffer_size is set to a small value, the shuffling will be more localized and may introduce some partial dependencies between neighboring elements.

4) Shuffling the dataset is particularly useful during training to ensure that the model sees a diverse and randomized order of examples, which can improve generalization and prevent the model from memorizing the order or patterns in the data.
'''
lines = tf.data.TextLineDataset(titanic_file)
counter1 = tf.data.Dataset.counter()

dataset = tf.data.Dataset.zip((counter1,lines))
dataset = dataset.shuffle(buffer_size = 100)
dataset = dataset.batch(20)
print(dataset)

# iterating over dataset:
n,line_batch = next(iter(dataset))
print(n.numpy())
'''
1) next(iter(dataset)): The iter(dataset) function returns an iterator over the elements of the dataset. The next() function is then used to get the next element from the iterator. This line retrieves the next element from the dataset.

2) n, line_batch = ...: The retrieved element from the dataset is assigned to two variables, n and line_batch, using tuple unpacking. The specific structure of the element depends on the structure of the dataset. Each element in the dataset can consist of multiple tensors or values, and here we assume that the element has two components: n and line_batch.

3) print(n.numpy()): n is a TensorFlow tensor object. By calling .numpy() on the tensor, we convert it into a NumPy array. This line prints the value of n as a NumPy array.
'''

'''
1) when iterating over the elements of the shuffled and repeated dataset, the elements are shuffled first, and then the shuffled order is repeated for the specified number of epochs 

2) Placing the shuffle() before the repeat() can be useful when you want to shuffle the dataset once and then repeat the shuffled order for multiple epochs during training. This ensures that the model sees a different shuffled order in each epoch, which can help with generalization and preventing the model from memorizing the order or patterns in the data.

'''

# Shuffle before repeat:- It will show every elem. of one epoch before moving to next.
dataset = tf.data.Dataset.zip((counter1,lines))
shuffled = dataset.shuffle(buffer_size = 100).batch(10).repeat(2)

print('Here are the item IDs near the epoch boundry: ')
for n,line_batch in shuffled.skip(60).take(5):
    print(n.numpy())

shuffle_repeat = [n.numpy().mean() for n,line_batch in shuffled]
plt.plot(shuffle_repeat,label = 'shuffle().repeat()')
plt.ylabel('Mean item ID')
plt.legend()
print()

'''
1) The zip() transformation is used to combine multiple datasets into a single dataset. In this case, counter1 and lines are two separate datasets, and zip((counter1, lines)) combines them element-wise into a new dataset. Each element of the resulting dataset will be a tuple containing an element from counter1 and an element from lines.

2) shuffle(buffer_size=100): This transformation shuffles the elements of the dataset randomly. The buffer_size parameter specifies the number of elements to load into a buffer for shuffling. In this case, it is set to 100.

3) batch(10): This transformation groups the elements of the dataset into batches of size 10. Each batch will contain 10 elements from the shuffled dataset.
   repeat(2): This transformation repeats the elements of the dataset for a specified number of epochs. In this case, it repeats the shuffled and batched dataset twice.
   shuffled.skip(60): The skip(60) transformation skips the first 60 elements of the dataset. This is useful to start iterating from a specific point in the dataset.
   take(5): The take(5) transformation selects the next 5 elements from the dataset. It limits the number of elements to be processed in the loop.

4) for n, line_batch in ...: In the loop, each element of the dataset is unpacked into n and line_batch. Here, n represents an item ID, and line_batch represents a batch of lines.

5) print(n.numpy()): This line prints the value of n (item ID) as a NumPy array.

6) [n.numpy().mean() for n, line_batch in shuffled]: This list comprehension iterates over the elements of the shuffled dataset, unpacks each element into n and line_batch, and calculates the mean value of n using n.numpy().mean(). The result is a list of mean item IDs from the shuffled dataset.

7) plt.plot(shuffle_repeat, label='shuffle().repeat()'): This line plots the list of mean item IDs (shuffle_repeat) on a graph. It labels the plot as 'shuffle().repeat()'.
   plt.ylabel('Mean item ID'): This line sets the label for the y-axis of the plot as 'Mean item ID'.
   plt.legend(): This line adds a legend to the plot.

8) A legend is a key that provides information about the elements represented in the plot. It helps to identify different data series or categories in the plot.
When multiple lines or data series are plotted on a graph, it can become difficult to differentiate between them visually. The legend serves as a visual guide that maps each line or series to a label, making it easier to understand and interpret the plot.
'''
# Shuffle after repeat:- 
dataset = tf.data.Dataset.zip((counter1,lines))
shuffled = dataset.repeat(2).shuffle(buffer_size = 100).batch(10)

print('Here are the item IDs near the epoch boundry: ')
for n,line_batch in shuffled.skip(55).take(15):
    print(n.numpy())

repeat_shuffle = [n.numpy().mean() for n,line_batch in shuffled]
plt.plot(shuffle_repeat,label = 'shuffle().repeat()')
plt.plot(repeat_shuffle,label = 'repeat().shuffle()')
plt.ylabel('Mean item ID')
plt.legend()
print()

'''
1) when iterating over the elements of the repeated and shuffled dataset, the dataset is first repeated three times, and then the repeated dataset is shuffled randomly. The order of the repeated elements is randomized within each repetition, but the repetitions themselves maintain their original order.

2) Placing the shuffle() after the repeat() can be useful when you want to repeat the dataset for multiple epochs before introducing randomness by shuffling. This can be helpful in scenarios where you want to ensure that the model sees the same data multiple times before shuffling to improve convergence and stability during training.

'''
# Preprocessing of Data:-
'''
1) In TensorFlow, the "map" transformation is a common method used in data preprocessing pipelines. It is primarily used to apply a specific function or operation to each element of a dataset, enabling you to transform and manipulate the data according to your requirements.

2) The map transformation in TensorFlow works by iterating over each element in the dataset and applying a given function to that element. This function can perform a wide range of operations, such as data normalization, feature extraction, data augmentation, or any custom transformation you want to apply to the data.

3) The map transformation is beneficial in data preprocessing pipelines as it allows you to perform various transformations efficiently on large datasets without explicitly iterating over each element
'''
# Decoding Image Data and resizing it:- Using map function
# We will be converting images of diff. sizes into same common size, so that they can be batched into a fixed size.
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

# reading an image from a file, decodes it into a dense tensor, and resizes it to a fixed image.
def parse_image(filename):
    parts = tf.strings.split(file_path, '/')
    label = parts[-2]
    
    image = tf.io.read_file(filename)                           # reading image from file,
    image = tf.image.decode_jpeg(image)                         # decoding image into jpeg form,
    image = tf.image.convert_image_dtype(image, tf.float32)     # converting datatype of image into float32,
    image = tf.image.resize(image,[128,128])                    # resizing image into (128X128) tensor,
    return image,label                                          # returning parsed image along with label.

file_path = next(iter(list_ds))
image,label = parse_image(file_path)

def show(image,label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')
    
show(image,label)

# Applying Arbitrary Python Logic: using tf.py_function()
# we want to apply a random rotation, the tf.image module only has tf.image.rot90, which is not very useful for image augmentation.
# we will use scipy.ndimage.rotate function instead.
import scipy.ndimage as ndimage

def random_rotate_image(image):
    image = ndimage.rotate(image,np.random.uniform(-30,30), reshape = False)
    return image

'''
1) image = ndimage.rotate(image, np.random.uniform(-30,30), reshape=False): This line performs the rotation on the input image. The ndimage.rotate function takes three main arguments: the image to rotate, the angle of rotation, and the reshape parameter. The np.random.uniform(-30, 30) generates a random angle between -30 and 30 degrees. The reshape=False parameter ensures that the output image retains the same shape as the input image, without any additional padding or cropping.

2) return image: This line returns the rotated image as the output of the function.

'''
image,label = next(iter(images_ds))
image = random_rotate_image(image)
show(image,label)

