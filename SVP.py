import numpy as np
A = np.array([[1, 2], [3, 4]])
u, s, vT = np.linalg.svd(A)
sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(sigma, s)
B = u @ sigma @ vT

print("orginal Matrix",A)
print("matrix",u)
print("singular value",s)
print("matrix sigma",sigma)
print("matrix vT",vT)
print("reconstructed matrix",B)

"""
OUTPUT
mlm@mlm-ThinkCentre-E73:~$ python3 singular.py
orginal Matrix [[1 2]
 [3 4]]
matrix [[-0.40455358 -0.9145143 ]
 [-0.9145143   0.40455358]]
singular value [5.4649857  0.36596619]
matrix sigma [[5.4649857  0.        ]
 [0.         0.36596619]]
matrix vT [[-0.57604844 -0.81741556]
 [ 0.81741556 -0.57604844]]
reconstructed matrix [[1. 2.]
 [3. 4.]]
"""
