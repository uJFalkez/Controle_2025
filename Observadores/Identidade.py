import numpy as np
from scipy.signal import place_poles
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def Observador_Identidade(A, B, C, K_LQR, POLOS, plot=True):
    # Ajuste dos polos do observador
    observer_poles = [a - 5 for a in POLOS]

    # Calcula o ganho L do observador via alocação de polos no sistema transposto
    place_obj = place_poles(A.T, C.T, observer_poles)
    L_id = place_obj.gain_matrix.T

    #import sympy as sp
    #print(sp.latex(sp.Matrix(L_id)))

    #para visualizar os polos do sistema observado, montaremos a matriz A_aug do sistema aumentado
    #print("Polos da planta:")
    #for pole in np.linalg.eigvals(A - B @ K_LQR):
    #    print(pole)
    
    #print("Polos do observador:")
    #for pole in np.linalg.eigvals(A - L_id @ C):
    #    print(pole)

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
    x0 = np.array([1, -2, 1, 1, 1, 1])  # estado real inicial
    xhat0 = np.zeros_like(x0)                   # estimativa inicial do observador
    z0 = np.concatenate([x0, xhat0])

    # Parâmetros de simulação
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Integração usando solve_ivp
    sol = solve_ivp(augmented_system, t_span, z0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2]  += np.pi/4
    sol.y[8]  += np.pi/4
    sol.y[4]  += np.pi/2
    sol.y[10] += np.pi/2

    mean_error_1 = sum([abs(v-e) for v, e in zip(sol.y[1], sol.y[7] )])/1000
    mean_error_2 = sum([abs(v-e) for v, e in zip(sol.y[3], sol.y[9])])/1000
    mean_error_3 = sum([abs(v-e) for v, e in zip(sol.y[5], sol.y[11])])/1000
    print("\nErros para observador ID")
    print(f"Erro médio para dx:",       mean_error_1)
    print(f"Erro médio para dtheta_1:", mean_error_2)
    print(f"Erro médio para dtheta_2:", mean_error_3)

    if plot:
        # Plot resultados
        plt.figure(figsize=(12, 6))
        for i, label in enumerate([r"$\dot{x}$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]):
            plt.plot(sol.t, sol.y[i*2+1, :], label=label)
            
        for i, label in enumerate([r"$\dot{x}$ obs.", r"$\dot{\theta}_1$ obs.", r"$\dot{\theta}_2$ obs"]):
            plt.plot(sol.t, sol.y[i*2+7, :], linestyle=":", label=label)
        
        plt.xlabel('Tempo [s]')
        plt.ylabel('Estado')
        plt.title('Estados reais vs estimados pelo observador identidade')
        plt.legend()
        plt.grid(True)
        plt.show()