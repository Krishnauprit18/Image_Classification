# Creating a dataset:-
# 1). a soyurce data constructs a dataset from data stored in memory or in files,
# 2). a data transformation constructs a dataset from one or more tf.data.dataset obj..
import matplotlib.pyplot as plt      
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([8,3,0,8,2,1])
print('dataset is: ',dataset)

for elem in dataset:
    print(elem.numpy()) # type: ignore
print()

iterate = iter(dataset)
print(next(iterate).numpy())
print()

# reducing dataset to its sum:-
print(dataset.reduce(0,lambda state,value: state+value).numpy())   # type: ignore
print()

# element specification:-
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4,10]))
print('specification of datset is: ',dataset1.element_spec)
print()

# Datasrt cnotaining a sparse tensor:-
dataset2 = tf.data.Dataset.from_tensor_slices(tf.SparseTensor(indices = [[0,0],[1,2]],values = [1,2],dense_shape = [3,4]))
print(dataset2)
print()
print(dataset2.element_spec.value_type) # type: ignore
print()

# some operations on dataset elem.:-
# Batching: it stacks n consecutive elem. of a dataset into single elem..
# All elem. must have a tensor of exactly same shape.
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0,-100,-1)
zipped_dataset = tf.data.Dataset.zip((inc_dataset,dec_dataset))
batched_dataset = dataset.batch(4)  # dataset of batch of 4 elem..

# iterating our batched dataset:
iterate = iter(batched_dataset)
for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])
print()

# Batching tensors with padding:- since above we worked with all tensors of same size. But models like sequence models 
# work with input data of varying size(seq. of diff. lengths).So to handle this case we use padded batch transformation.
# It enables us to batch tensors of different shape by specifying one or more dim. in which they are padded.

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x,tf.int32)],x))
dataset = dataset.padded_batch(4,padded_shapes = (None,))  

for batch in dataset.take(2):
    print(batch.numpy())             # type: ignore
    print()
    
    
''' 
1)dataset = tf.data.Dataset.range(100): This line creates a TensorFlow Dataset object called 
dataset using the tf.data.Dataset.range() function. This dataset contains a sequence of numbers from 0 to 99.

2) dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x)): The map() function is used to transform each element of the dataset. In this case, a lambda function is applied to each element x in the dataset. The lambda function uses tf.fill() to create a tensor filled with the value x. The shape of the tensor is determined by tf.cast(x, tf.int32), where x is cast to an integer. So, for each element in the original dataset, a tensor of shape [x, x, x, ..., x] is created.

3) dataset = dataset.padded_batch(4, padded_shapes=(None,)): The padded_batch() function is used to create batches from the dataset, where each batch is padded to the same shape. In this case, the batch size is set to 4. The padded_shapes argument specifies the desired shape of the padded elements in the batch. (None,) indicates that the first dimension (batch dimension) can vary, while the second dimension (the shape of each element) remains unspecified.   
'''
# It is also possible to override the padding val., which defaults to 0.
