import numpy as np
from scipy.signal import place_poles
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def Observador_Trivial(A, B, C, K_LQR, POLOS):
    # Ajuste dos polos do observador
    observer_poles = [a - 5 for a in POLOS]

    # Calcula o ganho L do observador via alocação de polos no sistema transposto
    place_obj = place_poles(A.T, C.T, observer_poles)
    L_id = place_obj.gain_matrix.T

    #print(L_id)

    # Sistema aumentado: estado real + estimativa
    def augmented_system(t, z):
        n = A.shape[0]
        x = z[:n]       # estado real
        x_hat = z[n:]   # estimativa do observador

        y = C @ x
        u = -K_LQR @ x_hat

        dxdt = A @ x + B @ u
        dx_hat_dt = A @ x_hat + B @ u + L_id @ (y - C @ x_hat)

        return np.concatenate([dxdt, dx_hat_dt])

    # Condição inicial
    x0 = np.array([0.1, 0, 0.05, 0, 0.02, 0])  # estado real inicial
    xhat0 = np.zeros_like(x0)                   # estimativa inicial do observador
    z0 = np.concatenate([x0, xhat0])

    # Parâmetros de simulação
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Integração usando solve_ivp
    sol = solve_ivp(augmented_system, t_span, z0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    # Plot resultados
    plt.figure(figsize=(12, 6))
    #for i, label in enumerate([r"$dx$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]):
    #    plt.plot(sol.t, sol.y[i*2+1, :], label=label)
        
    #for i, label in enumerate([r"$dx$ obs.", r"$\dot{\theta}_1$ obs.", r"$\dot{\theta}_2$ obs"]):
    #    plt.plot(sol.t, sol.y[i+6, :], linestyle=":", label=label)
    
    for i in range(9):
        plt.plot(sol.t, sol.y[i*2+1, :], label=f"x[{i}]")
    
    plt.xlabel('Tempo [s]')
    plt.ylabel('Estado')
    plt.title('Estados reais vs estimados pelo observador trivial')
    plt.legend()
    plt.grid(True)
    plt.show()