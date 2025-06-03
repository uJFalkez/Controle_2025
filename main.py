import sympy as sp
import copy

sin = sp.sin
cos = sp.cos
PI = sp.pi

SILENT = True

# Definição de Símbolos
x, theta1, theta2 = sp.symbols(r'x \tilde{\theta}_1 \tilde{\theta}_2')
dx, dtheta1, dtheta2 = sp.symbols(r'\dot{x} \dot{\tilde{\theta}}_1 \dot{\tilde{\theta}}_2')
d2x, d2theta1, d2theta2 = sp.symbols(r'\ddot{x} \ddot{\tilde{\theta}}_1 \ddot{\tilde{\theta}}_2')

g, L, m, M, F, Fp, T, Tp1, Tp2, I, alpha, beta = sp.symbols(r'g L m M \tilde{F} F_p \tilde{T} T_{p_1} T_{p_2} I \alpha \beta')

# Definição das equações do movimento
exp1 = 3*L*m*beta*d2theta1 + (4*m + 2*M)*d2x - 2*F - 2*Fp
exp2 = -6*g*L*m*alpha*theta1 + 5*L**2*m*d2theta1 - 2*L**2*m*alpha*d2theta2 + 6*L*m*beta*d2x + 4*I*d2theta1 - T - Tp1
exp3 = 2*g*L*m*theta2 - 2*L**2*m*alpha*d2theta1 + L**2*m*d2theta2 + 4*I*d2theta2 - Tp2

eq1 = sp.Eq(exp1, 0)
eq2 = sp.Eq(exp2, 0)
eq3 = sp.Eq(exp3, 0)

# Solução do sistema para d2x, d2theta1, d2theta2
A_, b = sp.linear_eq_to_matrix([eq1.lhs, eq2.lhs, eq3.lhs], [d2x, d2theta1, d2theta2])

sol = A_.LUsolve(-b)

d2x_expr, d2theta1_expr, d2theta2_expr = sol[0].simplify(), sol[1].simplify(), sol[2].simplify()

# Derivação do espaço de estados
dXdt = sp.Matrix([
    dx,             # d(x)/dt
    d2x_expr,       # d²(x)/dt²
    dtheta1,        # d(theta1)/dt
    d2theta1_expr,  # d²(theta1)/dt²
    dtheta2,        # d(theta2)/dt
    d2theta2_expr   # d²(theta2)/dt²
])

X_vetor = sp.Matrix([x, dx, theta1, dtheta1, theta2, dtheta2])
u_vetor = sp.Matrix([F, T])
p_vetor = sp.Matrix([Fp, Tp1, Tp2])

A = dXdt.jacobian(X_vetor).applyfunc(sp.simplify)
B = dXdt.jacobian(u_vetor).applyfunc(sp.simplify)
E = dXdt.jacobian(p_vetor).applyfunc(sp.simplify)

# Determinação dos valores numéricos
subs_dict = {
    L: 0.3,
    m: 0.06,
    M: 0.25,
    g: 9.81,
    I: 0.00045,
    alpha: 0.707,
    beta: 0.707
}

A_num = A.applyfunc(lambda exp: exp.subs(subs_dict))
B_num = B.applyfunc(lambda exp: exp.subs(subs_dict))
E_num = E.applyfunc(lambda exp: exp.subs(subs_dict))

# Derivação das funções de transferência
C_matrix = sp.Matrix([[1,0,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0,0,0,0,1,0]])

I_matrix = sp.eye(6)

s = sp.Symbol('s')

import numpy as np
from scipy.signal import ss2tf

# 1) Converte as matrizes Sympy para NumPy:
A = np.array(A_num.tolist(), dtype=float)
B = np.array(B_num.tolist(), dtype=float)
E = np.array(E_num.tolist(), dtype=float)
C = np.array(C_matrix.tolist(), dtype=float)
D = np.zeros([3,3])   # D = zeros(3×3)

G_num_list = []
for j in range(E.shape[1]):
    # passa apenas a j-ésima coluna de E e D
    Ej = E[:, j:j+1]
    Dj = D[:, j:j+1]
    num_j, den = ss2tf(A, Ej, C, Dj)
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

print(sp.latex(G))

