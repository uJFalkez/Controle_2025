import numpy as np
from scipy import signal as sig

def Controlador_AP_Pert(A, B, POLOS):
    # Todos os polos se encontram no eixo imaginário. Alocaremos todos 3 unidades reais para a esquerda:
    p_aloc = [x - 3 for x in POLOS]

    # Matriz K do ganho para a alocação
    K = sig.place_poles(A, B, p_aloc).gain_matrix

    t_span = (0,10)
    t_eval = np.linspace(*t_span, 1000)

    # Perturbações
    dist_vals = np.random.uniform(-0.2, 0.2, size=(6, len(t_eval)))

    from scipy.interpolate import interp1d
    dist_set = [interp1d(t_eval, dist_vals[i], kind='zero', fill_value='extrapolate') for i in range(6)]

    # Dinâmica do sistema com realimentação u = -Kx
    def system(t, x):
        u = -K @ x

        # perturbações
        dist = np.array([dist_set[i](t) for i in range(6)])
        
        dxdt = A @ x + B @ u + dist
        return dxdt

    # Condição inicial (perturbação inicial pequena)
    x0 = np.array([0.1,0,0.05,0,np.pi,5])

    from scipy.integrate import solve_ivp
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    from matplotlib import pyplot as plt
    # Plotar apenas os estados de posição: x1, x3 e x5
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], label='$x$ (m)')
    plt.plot(sol.t, sol.y[2], label=r'$\theta1$ (rad)')
    plt.plot(sol.t, sol.y[4], label=r'$\theta2$ (rad)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Posições')
    plt.title('Resposta das posições do sistema não linear com realimentação de estados')
    plt.grid(True)
    plt.legend()
    plt.show()