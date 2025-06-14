import numpy as np
def Observabilidade(A, C):
    AT = A.T
    CT = C.T
    N = np.hstack([np.linalg.matrix_power(AT, i) @ CT for i in range(6)])
    rank = np.linalg.matrix_rank(N)
    print(rank)
    # Printa 6 :D
    return rank