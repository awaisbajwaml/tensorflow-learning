import numpy as np

def printNumpyData(label, rawArray):
    numpyArray = np.array(rawArray)
    #Number of array dimensions.
    print(label)
    print(rawArray)
    print("Number of Array Dimensions ", numpyArray.ndim)
    #Tuple of integers giving the size of the array along each dimension
    #This means len(shape) = numpyArray.ndim
    #each dimension is a "subarray
    print("Shape: ", numpyArray.shape)
    print("           ")

# numpy is a library for scientific computing
# for example it can be used to represent matrices and perform operations on them
# a matrix can be represented as a multidimensional array
# the numpy array is a multidimensional array object per default
# you can consider each further dimension to be a an "array of array"

#start with one element array - the dimension is 1, the length of the array is 1
#the shape is the tuple of integers giving the size of the array along each dimension
#as the array has only one dimension - the shape should also be of length 1 (see definition above)

#initial
label = "2 elements on dimension 1"
rawArray = [1,2]
printNumpyData(label, rawArray)

#add elements
label = "4 elements on dimension 1"
rawArray = [1,2,3,4]
printNumpyData(label, rawArray)

#add dimension
label = "2 element on dimension 1, 1 element on dimension 2"
rawArray = [[1], [2]]
printNumpyData(label, rawArray)

#add elements
label = "2 element on dimension 1, 4 element on dimension 2"
rawArray = [[1, 2 ,3 ,4], [1, 2, 3 , 4]]
printNumpyData(label, rawArray)

#add dimension
label = "2 element on dimension 1, 1 element on dimension 2, 1 element on dimension 3"
rawArray = [[[1]], [[2]]]
printNumpyData(label, rawArray)

#add element
label = "2 element on dimension 1, 2 element on dimension 2, 1 element on dimension 3"
rawArray = [[[1], [2]], [[3], [4]]]
printNumpyData(label, rawArray)

#add element
label = "2 element on dimension 1, 2 element on dimension 2, 2 element on dimension 3"
rawArray = [[[1, 2], [3, 4]], [[4, 5], [5, 6]]]
printNumpyData(label, rawArray)


