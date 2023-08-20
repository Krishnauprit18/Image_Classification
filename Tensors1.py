import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
# loading data:
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# (x_trian,y_train) are features and labels of train dataset and (x_test and y_test) are features and labels of test dataset.
print('Attribute of training data tensor:')
print('=============================================================')
print('shape of x_train: ',x_train.shape) # 60000 elem. each of size 28X28
print('dim of x_train: ',x_train.ndim)
print('data type of x_train: ',x_train.dtype) # each elem. has 8 bit integer between 0-255
print()
print('Attributes of training label tensor: ')
print('==============================================================')
print('shape of y_train: ',y_train.shape)
print('dim of y_train: ',y_train.ndim)
print('data type of y_train: ',y_train.dtype)
print()
print('Attributes of testing data tensor: ')
print('==============================================================')
print('shape of x_test: ',x_test.shape)
print('dim of x_test: ',x_test.ndim)
print('data type of x_test: ',x_test.dtype)
print()
print('Attributes of testing label tensor: ')
print('shape of y_test: ',y_test.shape)
print('dim of y_test: ',y_test.ndim)
print('data type of y_test: ',y_test.dtype)
print()

# selecting specific elem. in tensor by Tensor Slicing:-
# selecting 1st data point(1st image) from training tensor:
x1 = x_train[0]
print('1st image is: ',x1)

def disp_image(img):
    plt.imshow(img,cmap = plt.cm.binary) # type: ignore
    plt.show()
    
disp_image(x1)
print()

###########################################################################

x2 = x_train[10]
print('11th data point is: ',x2)

def disp_image1(img):
    plt.imshow(img,cmap = plt.cm.binary) # type: ignore
    plt.show()
    
disp_image1(x2)
print()

##########################################################################

# selecting multiple datapts.:
x_train_slice = x_train[10:100]
print(x_train_slice)
print()
print('Attributes of datapoints from 10 to 100: ')
print('===================================================')
print('shape of datapt. is: ',x_train_slice.shape)
print('dimension of datapt. is: ',x_train_slice.ndim)
print('datatype of datapt. is: ',x_train_slice.dtype)
print()

########################################################################

# Q1). Select buttom right patch of 14X14 from training images.
x_train_br_slice = x_train[:,14:,14:]
print('Attributes of data sliced tensor is: ')
print('====================================================================')
print('shape of sliced tensor is: ',x_train_br_slice.shape)
print('dim of sliced tensor is: ',x_train_br_slice.ndim)
print('dtype of sliced tensor is: ',x_train_br_slice.dtype)
print()

#########################################################################

# Q2). Crop images of patches of 14X14 pixel centered in the middle.
x_train_br_slice1 = x_train[:,7:-7,7:-7]
print('Attributes of data sliced tensor is: ')
print('====================================================================')
print('shape of sliced tensor is: ',x_train_br_slice1.shape)
print('dim of sliced tensor is: ',x_train_br_slice1.ndim)
print('dtype of sliced tensor is: ',x_train_br_slice1.dtype)
print()

# Data Batches: We break data into batches and process those batches.
# 1st axis in data tensor is sample axis or sample dim.
# 1st axis in batch tensor is batch axis and batch dim.
# First batch:- first 128 exmaples. Each batch has 128 examples.
first_batch = x_train[:128]
print('Attributes of 1st batch sliced data tensor is: ')
print('====================================================================')
print('shape of sliced tensor is: ',first_batch.shape)
print('dim of sliced tensor is: ',first_batch.ndim)
print('dtype of sliced tensor is: ',first_batch.dtype)
print()

##########################################################################

# Second Batch:- second 128 examples.Each batche has next 128 examples.
second_batch = x_train[128:256]
print('Attributes of 2nd batch sliced data tensor is: ')
print('====================================================================')
print('shape of sliced tensor is: ',second_batch.shape)
print('dim of sliced tensor is: ',second_batch.ndim)
print('dtype of sliced tensor is: ',second_batch.dtype)
print()

###########################################################################

 