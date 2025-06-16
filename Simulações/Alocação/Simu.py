import numpy as np
from scipy import signal as sig

def Controlador_AP(A, B, POLOS):
    # Todos os polos se encontram no eixo imaginário. Alocaremos todos 3 unidades reais para a esquerda:
    p_aloc = [x - 3 for x in POLOS]

    # Matriz K do ganho para a alocação
    K = sig.place_poles(A, B, p_aloc).gain_matrix
    
    #print("Polos de malha fechada:", np.linalg.eigvals(A - B @ K))
    
    # Dinâmica do sistema com realimentação u = -Kx
    def system(t, x):
        u = -K @ x
        dxdt = A @ x + B @ u
        return dxdt

    # Condição inicial (perturbação inicial pequena)
    x0 = np.array([1, -2, 1, 1, 1, 1])

    # Simular de t=0s a t=5s
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    from scipy.integrate import solve_ivp
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    from matplotlib import pyplot as plt

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
    plt.title('Controle Alocação de Polos')
    plt.grid(True)
    plt.show()