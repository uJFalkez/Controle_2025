import sympy as sp

# Variável de Laplace
s = sp.symbols('s')

# Define matrizes A, B, C, D (exemplo)
A = sp.Matrix([[0, 1], [-2, -3]])
B = sp.Matrix([[0], [1]])
C = sp.Matrix([[1, 0]])
D = sp.Matrix([[0]])

# Identidade
I = sp.eye(A.shape[0])

# Função de Transferência G(s)
G = C * (s*I - A).inv() * B + D

# Simplifica a FT
G_simplificada = sp.simplify(G[0])

print(G_simplificada)