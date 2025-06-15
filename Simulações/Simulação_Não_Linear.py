import sympy as sp

# --- 1. Definir símbolos ---
t = sp.symbols('t')

# Coordenadas generalizadas (funções do tempo)
x = sp.Function('x')(t)
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)

# Derivadas temporais
dx = x.diff(t)
dtheta1 = theta1.diff(t)
dtheta2 = theta2.diff(t)

# Parâmetros do sistema
M, m, L, g = sp.symbols('M m L g')

# Momento de inércia de barra em relação ao CM
I = (1/12)*m*L**2

# --- 2. Posições dos centros de massa ---
x_c1 = (L/2)*sp.cos(theta1)
y_c1 = x + (L/2)*sp.sin(theta1)

x_j = L*sp.cos(theta1)
y_j = x + L*sp.sin(theta1)

x_c2 = x_j + (L/2)*sp.cos(theta2)
y_c2 = y_j + (L/2)*sp.sin(theta2)

# --- 3. Velocidades dos centros de massa (derivadas temporais) ---
dx_c1 = x_c1.diff(t)
dy_c1 = y_c1.diff(t)

dx_c2 = x_c2.diff(t)
dy_c2 = y_c2.diff(t)

# --- 4. Energia cinética ---
T_block = (1/2)*M*dx**2
T_bar1 = (1/2)*m*(dx_c1**2 + dy_c1**2) + (1/2)*I*dtheta1**2
T_bar2 = (1/2)*m*(dx_c2**2 + dy_c2**2) + (1/2)*I*dtheta2**2

T = T_block + T_bar1 + T_bar2
T = sp.simplify(T)

# --- 5. Energia potencial ---
V = m*g*y_c1 + m*g*y_c2 + M*g*x
V = sp.simplify(V)

# --- 6. Lagrangiano ---
Lagr = T - V

# --- 7. Variáveis generalizadas e suas derivadas ---
q = sp.Matrix([x, theta1, theta2])
dq = sp.Matrix([dx, dtheta1, dtheta2])

# --- 8. Calcular as equações de movimento pelo Lagrange ---

eqs = []

for i in range(3):
    dL_dqi = sp.diff(Lagr, q[i])
    dL_ddqi = sp.diff(Lagr, dq[i])
    ddt_dL_ddqi = sp.diff(dL_ddqi, t)
    eq = ddt_dL_ddqi - dL_dqi
    eq = sp.simplify(eq)
    eqs.append(eq)

'''# --- 9. Mostrar as equações ---
for i, eq in enumerate(eqs):
    print(f"Equação {i+1}:")
    print(sp.latex(eq))
    print("\n---\n")'''

ddx, ddtheta1, ddtheta2 = sp.symbols('ddx ddtheta1 ddtheta2')

from sympy import linear_eq_to_matrix

# Substituir as segundas derivadas nas equações
eqs_sub = []
for eq in eqs:
    eq_sub = eq.replace(sp.Derivative(x, (t, 2)), ddx)
    eq_sub = eq_sub.replace(sp.Derivative(theta1, (t, 2)), ddtheta1)
    eq_sub = eq_sub.replace(sp.Derivative(theta2, (t, 2)), ddtheta2)
    eqs_sub.append(eq_sub)

# Obter a matriz A e vetor b
A, b = linear_eq_to_matrix(eqs_sub, [ddx, ddtheta1, ddtheta2])

# Resolver para as acelerações
sol = A.LUsolve(b)

ddx_expr, ddtheta1_expr, ddtheta2_expr = sol[0], sol[1], sol[2]

subs_dict = {
    L: 0.3,
    m: 0.15,
    M: 0.3,
    g: 9.81
}

ddx_num = ddx_expr.subs(subs_dict)
ddtheta1_num = ddtheta1_expr.subs(subs_dict)
ddtheta2_num = ddtheta2_expr.subs(subs_dict)

import numpy as np
from sympy import lambdify

vars_state = (x, x.diff(), theta1, theta1.diff(), theta2, theta2.diff())

ddx_func = lambdify(vars_state, ddx_num, 'numpy')
ddtheta1_func = lambdify(vars_state, ddtheta1_num, 'numpy')
ddtheta2_func = lambdify(vars_state, ddtheta2_num, 'numpy')

def ode_func(t, X):
    x_, dx_, th1, dth1, th2, dth2 = X
    ddx = ddx_func(x_, dx_, th1, dth1, th2, dth2)
    ddth1 = ddtheta1_func(x_, dx_, th1, dth1, th2, dth2)
    ddth2 = ddtheta2_func(x_, dx_, th1, dth1, th2, dth2)
    return [dx_, ddx, dth1, ddth1, dth2, ddth2]

from scipy.integrate import solve_ivp

PI = sp.pi
X0 = [0, 0, 0, 5, 0, 0]  # ajuste como quiser

t_final = 100
t_eval = np.linspace(0, t_final, 10000)

sol = solve_ivp(ode_func, [0, t_final], X0, t_eval=t_eval, method='RK45', max_step=0.01)

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0], label='x (bloco)')
plt.plot(sol.t, sol.y[2], label='theta1')
plt.plot(sol.t, sol.y[4], label='theta2')
plt.legend()
plt.xlabel('Tempo (s)')
plt.ylabel('Estado')
plt.title('Simulação do sistema não linear')
plt.grid()
plt.show()
