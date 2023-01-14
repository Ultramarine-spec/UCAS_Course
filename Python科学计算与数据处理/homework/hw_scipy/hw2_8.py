import numpy as np
from scipy import sparse

A = np.array([[3, 0, 8, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

B = sparse.dok_matrix(A)
print("dok_matrix:")
print(dict(B))
C = sparse.lil_matrix(A)
print("lil_matrix")
print("row:", C.rows)
print("data:", C.data)
D = sparse.coo_matrix(A)
print("coo_matrix")
print("row:", D.row)
print("col:", D.col)
print("data:", D.data)
