import numpy as np
# 0 dim tensor:
x = np.array(12)
print(x)
print()
print('shape is: ',x.shape)
print()
print('dim is: ',x.ndim)
print()
print('datatype is: ',x.dtype)
print()

# 1 dim tensor:
y = np.array([1,2,4,5,6])
print(y)
print()
print('dim is: ',y.ndim)
print()

# 2 dim tensor(matrix):
z = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print(z)
print()
print('dim is: ',z.ndim)
print()
print('shape is: ',z.shape)
print()
print('datatype is: ',z.dtype)
print()

u = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
u1 = np.array([[[1,2,3]],[[4,5,6]]])
print(u)
print()
print(u1)
print('shape is: ',u.shape)
print()
print('dim is: ',u.ndim)
print()