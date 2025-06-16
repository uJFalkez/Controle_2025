import numpy as np
import sympy as sp
def Observabilidade(A, C):
    AT = A.T
    CT = C.T
    N = np.hstack([np.linalg.matrix_power(AT, i) @ CT for i in range(6)])
    print(sp.latex(sp.Matrix(N)))
    rank = np.linalg.matrix_rank(N)
    print(rank)
    # Printa 6 :D
    return rank