import numpy as np
import sympy as sp
def Controlabilidade(A, B):
    Q = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(6)])
    print(sp.latex(sp.Matrix(Q)))
    rank = np.linalg.matrix_rank(Q)
    print(rank)
    # Printa 6 :D
    return rank