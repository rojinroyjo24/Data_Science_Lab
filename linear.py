import numpy as np
ar=np.array([[1,2],[3,4],])
print("Determinent")
print(np.linalg.det(ar))

print("inverse:")
print(np.linalg.inv(ar))

print("rank:")
print(np.linalg.matrix_rank(ar))

print("1d array:")
print(ar.flatten())

"""
OUTPUT
Determinent
-2.0000000000000004
inverse:
[[-2.   1. ]
 [ 1.5 -0.5]]
rank:
2
1d array:
[1 2 3 4]
"""
