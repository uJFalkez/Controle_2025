import numpy as np
from scipy.linalg import solve_continuous_are

def Controlador_LQR_Pert(A, B):
    Q = np.eye(A.shape[0])  # Peso unitário para todos os estados
    R = np.eye(B.shape[1])  # Peso unitário para todas as entradas de controle
    
    # Resolve a equação de Riccati contínua
    P = solve_continuous_are(A, B, Q, R)

    # Calcula K
    K = np.linalg.inv(R) @ B.T @ P
    # print(sp.latex(K))

    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    # Intervalo de simulação
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Perturbações
    dist_vals = np.random.uniform(-0.2, 0.2, size=(6, len(t_eval)))

    from scipy.interpolate import interp1d
    dist_set = [interp1d(t_eval, dist_vals[i], kind='zero', fill_value='extrapolate') for i in range(6)]

    def sistema(t, x):
        u = -K @ x
        dist = np.array([dist_set[i](t) for i in range(6)])
        dxdt = A @ x + B @ u + dist
        return dxdt

    # Estado inicial
    x0 = np.array([1, -2, 1, 1, 1, 1])

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
    plt.grid(True)
    plt.show()