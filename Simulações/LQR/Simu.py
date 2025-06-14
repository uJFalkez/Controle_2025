import numpy as np
from scipy.linalg import solve_continuous_are

from Matrizes import *

def Controlador_LQR(A, B):
    Q = np.eye(A.shape[0])  # Peso unitário para todos os estados
    R = np.eye(B.shape[1])  # Peso unitário para todas as entradas de controle

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

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

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