import numpy as np
from scipy.linalg import solve_continuous_are

def Controlador_LQR_Pert(A, B, E):
    max_states = 1.5, 2, 1, 1, 1, 1
    max_inputs = 0.8, 1
    
    Q = np.diag([1/m**2 for m in max_states])
    
    R = np.diag([1/n**2 for n in max_inputs])
    
    # Resolve a equação de Riccati contínua
    P = solve_continuous_are(A, B, Q, R)

    # Calcula K
    K = np.linalg.inv(R) @ B.T @ P
    #print(K)

    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    # Intervalo de simulação
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Perturbações
    dist_vals = np.random.uniform(-0.05, 0.05, size=(3, len(t_eval)))

    from scipy.interpolate import interp1d
    dist_set = [interp1d(t_eval, dist_vals[i], kind='zero', fill_value='extrapolate') for i in range(3)]

    def sistema(t, x):
        u = -K @ x
        dist = np.array([dist_set[i](t) for i in range(3)])
        
        dist[0] += -0.002*x[1]**2
        dist[1] += -0.005*x[3]**2
        dist[2] += -0.005*x[5]**2
        
        dxdt = A @ x + B @ u + E @ dist
        return dxdt

    # Estado inicial
    x0 = np.array([1, -2, 1, 1, 1, 1])

    # Resolve a EDO
    sol = solve_ivp(sistema, t_span, x0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    # Plot das variáveis de estado
    plt.figure(figsize=(14,9))
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
    plt.title('Controle LQR + Perturbações')
    plt.grid(True)
    plt.show()