import numpy as np
from scipy.signal import place_poles
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def Observador_OR(A, B, C, K_LQR, plot=True):
    # Índices dos estados medidos e não medidos
    measured_idx = [0, 2, 4]   # posições medidas: x, theta1, theta2
    unmeasured_idx = [1, 3, 5] # velocidades não medidas

    # Dividir A em blocos
    A11 = A[np.ix_(measured_idx, measured_idx)]
    A12 = A[np.ix_(measured_idx, unmeasured_idx)]
    A21 = A[np.ix_(unmeasured_idx, measured_idx)]
    A22 = A[np.ix_(unmeasured_idx, unmeasured_idx)]

    # Dividir B em blocos
    B1 = B[measured_idx, :]
    B2 = B[unmeasured_idx, :]

    # Polos para observador reduzido
    observer_poles_reduced = np.array([-5, -6+7.5j, -6-7.5j])
    place_obj = place_poles(A22.T, A12.T, observer_poles_reduced)
    L_or = place_obj.gain_matrix.T  # 3x3

    #import sympy as sp
    #print(sp.latex(sp.Matrix(L_or)))

    #para visualizar os polos do sistema observado, montaremos a matriz A_aug do sistema aumentado
    #print("Polos da planta:")
    #for pole in np.linalg.eigvals(A - B @ K_LQR):
    #    print(pole)
    
    #print("Polos do observador:")
    #for pole in np.linalg.eigvals(A22 - L_or @ A12):
    #    print(pole)

    # Variável global para armazenar y anterior para diferença finita
    global y_prev, t_prev
    y_prev = None
    t_prev = None

    def system(t, z):
        global y_prev, t_prev

        # Separar estados reais e estimados
        x = z[:6]
        x_u_hat = z[6:]  # estimativa dos estados não medidos

        # Saída medida
        y = C @ x

        # Estimativa da derivada de y via diferença finita
        if y_prev is None or t_prev is None:
            y_dot = np.zeros_like(y)
        else:
            dt = t - t_prev
            if dt == 0:
                y_dot = np.zeros_like(y)
            else:
                y_dot = (y - y_prev) / dt

        # Atualiza valores anteriores
        y_prev = y
        t_prev = t

        # Vetor estimado completo
        x_hat = np.zeros(6)
        x_hat[measured_idx] = y         # parte medida
        x_hat[unmeasured_idx] = x_u_hat # parte estimada

        # Controle LQR com estado estimado completo
        u = -K_LQR @ x_hat

        # Dinâmica dos estados reais
        dxdt = A @ x + B @ u

        # Dinâmica do observador de ordem reduzida para estimar x_u_hat
        dx_u_hat_dt = (
            A22 @ x_u_hat + A21 @ y + B2 @ u + L_or @ (y_dot - A11 @ y - A12 @ x_u_hat - B1 @ u)
        )

        return np.concatenate([dxdt, dx_u_hat_dt])

    # Condições iniciais
    x0 = np.array([1, -2, 1, 1, 1, 1])
    x_u_hat0 = np.zeros(3)

    z0 = np.concatenate([x0, x_u_hat0])

    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    sol = solve_ivp(system, t_span, z0, t_eval=t_eval, vectorized=False)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    mean_error_1 = sum([abs(v-e) for v, e in zip(sol.y[1], sol.y[6])])/1000
    mean_error_2 = sum([abs(v-e) for v, e in zip(sol.y[3], sol.y[7])])/1000
    mean_error_3 = sum([abs(v-e) for v, e in zip(sol.y[5], sol.y[8])])/1000
    print("\nErros para observador OR")
    print(f"Erro médio para dx:",       mean_error_1)
    print(f"Erro médio para dtheta_1:", mean_error_2)
    print(f"Erro médio para dtheta_2:", mean_error_3)

    if plot:
        # Plot resultados
        plt.figure(figsize=(12, 6))
        for i, label in enumerate([r"$\dot{x}$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]):
            plt.plot(sol.t, sol.y[i*2+1, :], label=label)
            
        for i, label in enumerate([r"$\dot{x}$ obs.", r"$\dot{\theta}_1$ obs.", r"$\dot{\theta}_2$ obs"]):
            plt.plot(sol.t, sol.y[i+6, :], linestyle=":", label=label)
        
        plt.xlabel('Tempo [s]')
        plt.ylabel('Estado')
        plt.title('Estados reais vs estimados pelo observador de ordem reduzida')
        plt.legend()
        plt.grid(True)
        plt.show()