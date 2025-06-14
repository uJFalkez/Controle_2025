import numpy as np
from scipy.signal import place_poles
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def Observador_OR(A, B, C, K_LQR):
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

    # Polos para observador reduzido (escolha poles mais rápidos que controlador)
    observer_poles_reduced = np.array([-5, -6+7.5j, -6-7.5j])
    place_obj = place_poles(A22.T, A12.T, observer_poles_reduced)
    L_or = place_obj.gain_matrix.T  # 3x3

    # Variável global para armazenar y anterior para diferença finita
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
    x0 = np.array([0.1, 0, 0.05, 0, 0.02, 0])
    x_u_hat0 = np.zeros(3)

    z0 = np.concatenate([x0, x_u_hat0])

    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Reset variáveis globais antes da simulação
    y_prev = None
    t_prev = None

    sol = solve_ivp(system, t_span, z0, t_eval=t_eval, vectorized=False)

    # ajusta as soluções pros valores reais

    # Plot dos estados reais e estimados das velocidades (não medidas)
    plt.figure(figsize=(12, 6))
    for i, idx_real in enumerate(unmeasured_idx):
        plt.plot(sol.t, sol.y[idx_real, :], label=f'Real x{idx_real+1} (velocidade)')
        plt.plot(sol.t, sol.y[6 + i, :], '--', label=f'Estimado x{idx_real+1} (velocidade)')

    plt.xlabel('Tempo [s]')
    plt.ylabel('Velocidade')
    plt.title('Observador de ordem reduzida: estados reais e estimados')
    plt.legend()
    plt.grid()
    plt.show()
