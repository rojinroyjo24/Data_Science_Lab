import numpy as np
ar1=np.array([[1,2],[3,4]])
ar2=np.array([[5,6],[7,8]])
print("matrix addition")
print(np.add(ar1,ar2))

print("matrix sub")
print(np.add(ar1,ar2))

print("matrix addition")
print(np.subtract(ar1,ar2))

print("matrix addition")
print(np.add(ar1,ar2))

print("matrix multiply")
print(np.multiply(ar1,ar2))

print("matrix divide")
print(np.add(ar1,ar2))

print("matrix mul")
print(np.dot(ar1,ar2))

print("matrix trans")
print(ar1.transpose())

print("matrix diag")
print(np.trace(ar1))
"""
OUTPUT

mlm@mlm-ThinkCentre-E73:~/ds$ python3 matrix.py
matrix addition
[[ 6  8]
 [10 12]]
matrix sub
[[ 6  8]
 [10 12]]
matrix addition
[[-4 -4]
 [-4 -4]]
matrix addition
[[ 6  8]
 [10 12]]
matrix multiply
[[ 5 12]
 [21 32]]
matrix divide
[[ 6  8]
 [10 12]]
matrix mul
[[19 22]
 [43 50]]
matrix trans
[[1 3]
 [2 4]]
matrix diag
5
"""

