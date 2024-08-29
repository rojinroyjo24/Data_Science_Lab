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
