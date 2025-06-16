import numpy as np
from scipy import signal as sig

def Controlador_AP_Pert(A, B, E, POLOS):
    # Todos os polos se encontram no eixo imaginário. Alocaremos todos 3 unidades reais para a esquerda:
    p_aloc = [x - 3 for x in POLOS]

    # Matriz K do ganho para a alocação
    K = sig.place_poles(A, B, p_aloc).gain_matrix

    t_span = (0,10)
    t_eval = np.linspace(*t_span, 1000)

    # Perturbações
    dist_vals = np.random.uniform(-0.05, 0.05, size=(3, len(t_eval)))

    from scipy.interpolate import interp1d
    dist_set = [interp1d(t_eval, dist_vals[i], kind='zero', fill_value='extrapolate') for i in range(3)]

    # Dinâmica do sistema com realimentação u = -Kx
    def system(t, x):
        u = -K @ x

        # perturbações
        dist = np.array([dist_set[i](t) for i in range(3)])
        
        dist[0] += -0.002*x[1]**2
        dist[1] += -0.005*x[3]**2
        dist[2] += -0.005*x[5]**2
        
        dxdt = A @ x + B @ u + E @ dist
        return dxdt

    # Condição inicial (perturbação inicial pequena)
    x0 = np.array([1, -2, 1, 1, 1, 1])

    from scipy.integrate import solve_ivp
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    from matplotlib import pyplot as plt

    plt.figure(figsize=(14,9))
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
    plt.title('Controle Alocação de Polos + Perturbação')
    plt.grid(True)
    plt.show()