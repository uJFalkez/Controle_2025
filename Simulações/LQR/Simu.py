import numpy as np
from scipy.linalg import solve_continuous_are

def Controlador_LQR(A, B):
    # Regra de Bryson: Q_ii = 1/x_ii², R_jj = 1/u_jj²
    max_states = 1.5, 2, 1, 1, 1, 1
    max_inputs = 0.8, 1
    
    Q = np.diag([1/m**2 for m in max_states])
    
    R = np.diag([1/n**2 for n in max_inputs])
    
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
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 600)

    # Resolve a EDO
    sol = solve_ivp(sistema, t_span, x0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    # Plot das variáveis de estado
    for i, label in enumerate(["x", r"\theta_1", r"\theta_2"]):
        plt.plot(sol.t, sol.y[i*2], label=rf"${label}$")
    
    #for i, label in enumerate([r"\dot{x}", r"\dot{\theta}_1", r"\dot{\theta}_2"]):
    #    plt.plot(sol.t, sol.y[i*2+1], alpha=0.5, label=rf"${label}$")

    plt.axhline(y=0, alpha=0.5, color='r', linestyle='--', label=r'$x_{eq}$')
    plt.axhline(y=np.pi/4, alpha=0.5, color='r', linestyle='--', label=r'$\theta_{1,eq}$')
    plt.axhline(y=np.pi/2, alpha=0.5, color='r', linestyle='--', label=r'$\theta_{2,eq}$')

    plt.legend()
    plt.xlabel('Tempo (s)')
    plt.ylabel('Estados')
    plt.title('Controle LQR')
    plt.grid(True)
    plt.show()