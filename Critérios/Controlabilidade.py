import numpy as np
def Controlabilidade(A, B):
    Q = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(6)])
    rank = np.linalg.matrix_rank(Q)
    print(rank)
    # Printa 6 :D
    return rank