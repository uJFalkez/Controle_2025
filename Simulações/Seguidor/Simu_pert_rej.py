import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import place_poles

def Seguidor_Pert_Rej(A, B, C, E, L_or, K_LQR, plot=True):
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
    
    B_pinv = np.linalg.pinv(B)

    # referência
    x_ref = np.array([0.7, 0, 3*np.pi/4, 0, 0, 0]) # pi*3/4 é 135 graus de diferença até o equilíbrio, ou seja, seria 180 graus na realidade
    u_eq  = -B_pinv @ (A @ x_ref)

    def control_with_ref(x_hat, d_hat):
        return u_eq - K_LQR @ (x_hat - x_ref) - B_pinv @ (E @ d_hat)

    # variáveis para estimar ẏ por diferença finita
    global y_prev, t_prev
    y_prev = None
    t_prev = None

    t_span = (0,10)
    t_eval = np.linspace(*t_span, 1000)

    # Perturbações
    dist_vals = np.random.uniform(-0.05, 0.05, size=(3, len(t_eval)))

    from scipy.interpolate import interp1d
    dist_set = [interp1d(t_eval, dist_vals[i], kind='zero', fill_value='extrapolate') for i in range(3)]

    def reduced_order_system(t, z):
        global y_prev, t_prev

        x        = z[:6]    # reais
        xu_hat   = z[6:9]   # estimados (3)
        d_hat    = z[9:12]  # perturbações estimadas

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

        u = control_with_ref(x_hat, d_hat)

        # perturbações
        dist = np.array([dist_set[i](t) for i in range(3)])
        
        dist[0] += -0.002*x[1]**2
        dist[1] += -0.005*x[3]**2
        dist[2] += -0.005*x[5]**2
        
        # dinâmica real
        dx     = A @ x + B @ u + E @ dist

        innov = ydot - (A11 @ y + A12 @ xu_hat + B1 @ u) - (E[meas,:] @ d_hat)
        
        dxu_hat = A22 @ xu_hat + A21 @ y + B2 @ u + L_or @ innov
        
        dd_hat = np.diag([0.01, 0.1, 0]) @ innov
        
        return np.hstack([dx, dxu_hat, dd_hat])

    # iniciais
    x0    = np.array([1, -2, 1, 1, 1, 1])
    xu0   = np.zeros(3)
    z0    = np.hstack([x0, xu0, np.zeros(3)])

    y_prev = None
    t_prev = None

    sol = solve_ivp(reduced_order_system, t_span, z0, t_eval=t_eval)

    # ajusta as soluções pros valores reais
    sol.y[2] += np.pi/4
    sol.y[4] += np.pi/2

    mean_error_t1 = sum([abs(e-np.pi) for e in sol.y[2][250:]])/750
    mean_error_t2 = sum([abs(e-np.pi/2) for e in sol.y[4][250:]])/750
    print("\nCom correção")
    print(f"Erro médio para theta_1:", mean_error_t1)
    print(f"Erro médio para theta_2:", mean_error_t2)

    if plot:
        # plota só as velocidades (estados não medidos) reais vs estimadas
        plt.figure(figsize=(14,9))
        for idx, label in enumerate([r"$x$", r"$\theta_1$", r"$\theta_2$"]):
            plt.plot(sol.t, sol.y[idx*2], label=label)
        plt.axhline(x_ref[0], color='gray', linestyle=':', label=r'$x$ ref')
        plt.axhline(x_ref[2]+np.pi/4, color='black', linestyle=':', label=r'$\theta_1$ ref')
        plt.axhline(x_ref[4]+np.pi/2, color='black', linestyle=':', label=r'$\theta_2$ eq')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posições (m ou rad)')
        plt.title('Posições: Seguidor + Observador OR + LQR + Perturbações + DOB')
        plt.legend()
        plt.grid(True)
        plt.show()

        # velocidades em janela separada
        plt.figure(figsize=(14,9))
        for idx, label in enumerate([r"$\dot{x}$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]):
            plt.plot(sol.t, sol.y[idx*2+1], label=label)
        for idx, label in enumerate([r"$\dot{x}$ est.", r"$\dot{\theta}_1$ est.", r"$\dot{\theta}_2$ est."]):
            plt.plot(sol.t, sol.y[idx+6], linestyle=":", label=label)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidades (m ou rad)')
        plt.title('Velocidades: Seguidor + Observador OR + LQR + Perturbações + DOB')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return mean_error_t1, mean_error_t2