# In NN we specify layers with:
#   tf.keras.layers.Dense(128, activation = 'relu') we have 128 units and relu activation function.
# this layer takes 2d tensor as ip. and returns another 2d tensor as an op.
# in backend this layers performs dot product between parameter vector/weights(w) and the input and adds bias vector(b) to it, so this becomes
# the linear combination and it is subjected to non linear activation like relu/sigmoid.
# output = relu(dot(w,input)+b)

import numpy as np
import matplotlib.pyplot as plt
import math

w = np.array([[1,0.5],[2,1]])
print('axes of w is: ',w.ndim)
print('shape of w is: ',w.shape)
print()

input = np.array([[1,2],[-1,2]])
print('axes of ip. is: ',input.ndim)
print('shape of ip. is: ',input.shape)
print()

b = np.array([-2.0,0.5])
print('axes of b is: ',b.ndim)
print('shape of b is: ',b.shape)
print()

# Applying Linear Combination:-
z = np.dot(w,input)+b       # Element wise addition.
print('=========================================================')
print('shape of w: ',w.shape)
print('shape of input is: ',input.shape)
print('shape of b is: ',b.shape)
print('==========================================================')
print('shape of z is: ',z.shape)
print('===========================================================')
print('linear combination is: ',z)
print()

# Applying Relu on non linear activation:-
output = np.maximum(0.,z)   # Element wise relu.
print('shape of output is: ',output.shape)
print()
print(output)
# relu makes nagative numbers 0.
# visualizing relu function:-
x = np.linspace(-10,10,100)
z = np.maximum(0.,x)
plt.plot(x,z)
plt.xlabel('X')
plt.ylabel('relu(x)')
print()

# visualizing sigmoid function:-
x = np.linspace(-10,10,100)
z = 1/(1+np.exp(-x))

plt.plot(x,z)
plt.xlabel('X')
plt.ylabel('sigmoid(x)')

# relu and tensor addition are elem. wise operations and can be parallelized.
# Broadcasting:- We add 2 tensors when their shapes differ..It is employes to make 2 tensors involved in the operations compatible.
# 1. axes are added to the smaller tensor to match dim with larger tensor.And these axes are known as broadcast axes.
# 2. the smaller tensor is repeated alongisde these new axes to match the full shape of larger tensor.
x_1 = np.random.rand(32,10)  # Matrix
x_2 = np.random.rand(10,)   # 1-D tensor.

print('shape of x_1: ',x_1.shape)
print('shape of x_2: ',x_2.shape)
print()

# Broadcasting:
# adding a new axes to x_2 to match dim with x_1:
x_2 = np.expand_dims(x_2,axis = 0) # type: ignore
print('new shape of x_2 after adding broadcast axis: ',x_2.shape)
print()

# Reapeting x_2 tensor 32 times on the new axes to make compatible with x_1:
x_2 = np.repeat(x_2,32,axis = 0)
print('new shape of x_2 after repeatation (after broadcasting): ',x_2.shape)
print()

# numpy implements broadcasting:-
x_1 = np.random.rand(32,10)
x_2 = np.random.rand(10,)
print('implementing broadcasting by numpy: ',(x_1+x_2).shape)
print()

# Reshaping:- Used to arrange rows and columns of tensor to match the shape of target tensor.It has same number of elem. as the initial tensor.
# It is used in data pre-processing.

t = np.array([[0.,1.],
              [2.,3.],
              [4.,5.]])
print(t)
print('shape of t is: ',t.shape)
print()

# reshaping the tensor:-
t = t.reshape((6,1))
print('reshaped t is: ',t)
print()
t = t.reshape((2,3))
print('one more time reshaped t is: ',t)
print()

# Transposition is a special case of reshaping:-rows->cols.  and  cols.->rows
v = np.zeros((300,20))
v = np.transpose(v)
print('initial v is: ',v)
print('transposed v is: ',v)
print('shape of v is: ',v.shape)