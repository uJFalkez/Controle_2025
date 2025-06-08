import sympy as sp
import numpy as np
import Matrizes as MTx

# Derivação das funções de transferência
C_matrix = sp.Matrix([[1,0,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0,0,0,0,1,0]])

I_matrix = sp.eye(6)

s = sp.Symbol('s')

import numpy as np
from scipy.signal import ss2tf

# 1) Converte as matrizes Sympy para NumPy:
A_ = np.array(MTx.A.tolist(), dtype=float)
B_ = np.array(MTx.B.tolist(), dtype=float)
C_ = np.array(MTx.C.tolist(), dtype=float)
D_ = np.array(MTx.D.tolist(), dtype=float)
E_ = np.array(MTx.E.tolist(), dtype=float)

G_num_list = []
for j in range(B_.shape[1]):
    # passa apenas a j-ésima coluna de E e D
    Bj = B_[:, j:j+1]
    Dj = D_[:, j:j+1]
    num_j, den = ss2tf(A_, Bj, C_, Dj)
    G_num_list.append(num_j)

deg_den = len(den) - 1
D = sum(round(den[k],2) * s**(deg_den - k) for k in range(deg_den))

# 2) para cada entrada j e saída i, monta N_{ij}(s)
rows = []
for i in range(3):            # 3 saídas
    row = []
    for j in range(len(G_num_list)):  # 2 entradas
        coeffs = G_num_list[j][i]     # vetor de coef s^(n) -> s^0
        deg_num = len(coeffs) - 1
        N = sum(round(coeffs[k],2) * s**(deg_num - k) for k in range(deg_num+1))
        row.append(N/D)
    rows.append(row)

G = sp.Matrix(rows).applyfunc(lambda e: e.evalf(3, chop=True))

den = G[0].as_numer_denom()[1]
poles = sp.roots(sp.Poly(den, s))
zeros = {
    "x": {
        "F": None,
        "T": None
        },
    "theta1": {
        "F": None,
        "T": None
        },
    "theta2": {
        "F": None,
        "T": None
        }
}

for i, exp in enumerate(G):
    coord = ("x", "theta1", "theta2")[i//2]
    input = ("F", "T")[i%2]
    num = exp.as_numer_denom()[0]
    zeros[coord][input] = sp.roots(sp.Poly(num, s))

'''print("Polos:")
print(sp.latex(poles))
print()

for coord, inputs in zeros.items():
    for input, roots in inputs.items():
        print(f"{coord} vs {input}:")
        print(sp.latex(roots))
        print()
    print()'''