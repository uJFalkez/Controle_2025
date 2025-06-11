import numpy as np
from scipy.linalg import solve_continuous_are
from Matrizes import *

# Suponha que você já tenha A, B definidos como arrays numpy
# Defina Q e R (exemplo simples)
Q = np.eye(A.shape[0])  # Peso unitário para todos os estados
R = np.eye(B.shape[1])  # Peso unitário para todas as entradas de controle

A = np.array(A, dtype=float)
B = np.array(B, dtype=float)
C = np.array(C, dtype=float)
D = np.array(D, dtype=float)
E = np.array(E, dtype=float)
Q = np.array(Q, dtype=float)
R = np.array(R, dtype=float)

# Resolve a equação de Riccati contínua
P = solve_continuous_are(A, B, Q, R)

# Calcula o ganho ótimo K
K = np.linalg.inv(R) @ B.T @ P
# print(sp.latex(K))

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Suponha que A, B, K já definidos
def sistema(t, x):
    u = -K @ x
    dxdt = A @ x + B @ u
    return dxdt

# Estado inicial (exemplo)
x0 = np.array([1, -2, 1, 1, 1, 1])  # ajuste conforme seu sistema

# Intervalo de simulação
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 600)

# Resolve a EDO
sol = solve_ivp(sistema, t_span, x0, t_eval=t_eval)

# Plot das variáveis de estado
for i, label in zip(range(sol.y.shape[0]), ("x", r"\dot{x}", r"\theta_1", r"\dot{\theta}_1", r"\theta_2", r"\dot{\theta}_2")):
    plt.plot(sol.t, sol.y[i], label=rf"${label}$")

# Salva um dump das variáveis para a simulação 3D
for name, series in {"x":sol.y[0], "theta1":sol.y[2], "theta2":sol.y[4]}.items():
    with open(f"3D sim/{name}.txt", 'w') as file:
        file.write(",".join(str(x) for x in series))

plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Estados')
plt.title('Resposta do sistema com controle LQR')
plt.show()