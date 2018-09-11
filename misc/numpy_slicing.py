import numpy as np



#Slicing and index bases access:
#Index access[i]: the dimension is reduced by one.
#Slicing acess[low:high]: preserves the dimension

#one dimension
my_array = np.array([0,1,2,3,4])
#select element "0"
print(my_array[0])
#select elements "0", "1", "2" as 1 dimensionaly array
print(my_array[0:3])

#two dimension
my_array = np.array([[0,1,2,3,4], [5,6,7,8,9]])
#select element "0"
print(my_array[0][0])
#select elements "0", "1", "2" as 1 dimensionaly array
print(my_array[0, 0:3])
#select elements "0", "1", "2" as array with original dimensions (2 in this case)
print(my_array[0:2, 0:3])

#three dimension
my_array = np.array([[[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14]]])
#select element "0"
print(my_array[0][0][0])
#select elements "0", "1", "2" as 1 dimensionaly array
print(my_array[0, 0, 0:3])
#select elements "0", "1", "2" as 2 dimensionaly array
print(my_array[0, 0:1, 0:3])
#select elements "0", "1", "2" as 3 dimensionaly array
print(my_array[0:1, 0:1, 0:3])