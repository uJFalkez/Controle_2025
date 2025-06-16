import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def Seguidor(A, B, C, L_or, K_LQR, plot=True):
    # índices dos estados medidos e não medidos
    meas = [0,2,4]
    unm  = [1,3,5]

    # particiona A,B em blocos
    A11 = A[np.ix_(meas, meas)]
    A12 = A[np.ix_(meas, unm)]
    A21 = A[np.ix_(unm, meas)]
    A22 = A[np.ix_(unm, unm)]
    B1  = B[meas, :]
    B2  = B[unm, :]

    # referência
    x_ref = np.array([0.7, 0, 3*np.pi/4, 0, 0, 0]) # pi*3/4 é 135 graus de diferença até o equilíbrio, ou seja, seria 180 graus na realidade
    B_pinv= np.linalg.pinv(B)
    u_eq  = -B_pinv @ (A @ x_ref)

    def control_with_ref(x_hat):
        return u_eq - K_LQR @ (x_hat - x_ref)

    # variáveis para estimar ẏ por diferença finita
    global y_prev, t_prev
    y_prev = None
    t_prev = None

    t_span = (0,10)
    t_eval = np.linspace(*t_span, 1000)

    def reduced_order_system(t, z):
        global y_prev, t_prev

        x        = z[:6]    # reais
        xu_hat   = z[6:]    # estimados (3)

        # medida
        y = C @ x

        # estima dy
        if y_prev is None:
            ydot = np.zeros_like(y)
        else:
            dt   = t - t_prev
            ydot = (y - y_prev)/dt if dt>0 else np.zeros_like(y)
        y_prev, t_prev = y, t

        # monta vetor estado estimado completo
        x_hat        = np.zeros(6)
        x_hat[meas]  = y
        x_hat[unm]   = xu_hat

        u = control_with_ref(x_hat)
        
        # dinâmica real
        dx     = A @ x + B @ u

        # dinâm. observador reduzido (3 estados não medidos)
        dxu_hat = ( A22 @ xu_hat
                + A21 @ y
                + B2  @ u
                + L_or @ ( ydot - A11 @ y - A12 @ xu_hat - B1 @ u )
                )

        return np.hstack([dx, dxu_hat])

    # iniciais
    x0 = np.array([1, -2, 1, 1, 1, 1])
    xu0   = np.zeros(3)
    z0    = np.hstack([x0, xu0])

    y_prev = None
    t_prev = None

    sol = solve_ivp(reduced_order_system, t_span, z0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    if plot:
        plt.figure(figsize=(10,5))
        
        plt.axhline(x_ref[0], color='black', linestyle=':', label=r'$x$ ref')
        plt.axhline(x_ref[2]+np.pi/4, color='black', linestyle=':', label=r'$\theta_1$ ref')
        plt.axhline(x_ref[4]+np.pi/2, color='black', linestyle=':', label=r'$\theta_2$ eq')

        for i, label in enumerate([r"$x$", r"$\theta_1$", r"$\theta_2$"]):
            plt.plot(sol.t, sol.y[i*2], label=f'{label}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posições (m ou rad)')
        plt.title('LQR + OR + Modelos Assumidos')
        plt.legend()
        plt.grid(True)
        plt.show()
