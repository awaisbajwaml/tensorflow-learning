import numpy as np

#numpy_dimension.py should be understood
#remember: the shape is the tuple of integers giving the size of the array along each dimension
def printArray(numpyArray):
    print("Number of Array Dimensions ", numpyArray.ndim)
    print("Shape: ", numpyArray.shape)
    print("Data: ", numpyArray)
    print(" ")

#one dimension
my_array = np.array([0,1,2,3,4,5])
printArray(my_array)

#reshape to the same dimension, len(shape) = 1
reshaped = my_array.reshape((6,))
printArray(reshaped)

#reshape to two dimension, len(shape) = 2
reshaped = my_array.reshape((1,6))
printArray(reshaped)

#also reshape to two dimension, but different distribution, len(shape) = 2
reshaped = my_array.reshape((2,3))
printArray(reshaped)

#also reshape to two dimension, but different distribution, len(shape) = 2
reshaped = my_array.reshape((6,1))
printArray(reshaped)

#declare two dimension
my_array = np.array([[0,1],[2,3],[4,5]])
printArray(my_array)

#reshape to one dimension
reshaped = my_array.reshape((6,))
printArray(reshaped)