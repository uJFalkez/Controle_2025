import sympy as sp

sin = sp.sin
cos = sp.cos
PI = sp.pi

# Definição de Símbolos
x, theta1, theta2 = sp.symbols(r'x \tilde{\theta}_1 \tilde{\theta}_2')
dx, dtheta1, dtheta2 = sp.symbols(r'\dot{x} \dot{\tilde{\theta}}_1 \dot{\tilde{\theta}}_2')
d2x, d2theta1, d2theta2 = sp.symbols(r'\ddot{x} \ddot{\tilde{\theta}}_1 \ddot{\tilde{\theta}}_2')

g, L, m, M, I, alpha, beta = sp.symbols(r'g L m M I \alpha \beta')

# Definição das equações do movimento
exp1 = (1/2)*(3*L*m*beta*d2theta1 + (4*m + 2*M)*d2x)
exp2 = (1/4)*(-6*g*L*m*alpha*theta1 + 5*L**2*m*d2theta1 - 2*L**2*m*alpha*d2theta2 + 6*L*m*beta*d2x + 4*I*d2theta1)
exp3 = (1/4)*(-2*g*L*m*theta2 - 2*L**2*m*alpha*d2theta1 + L**2*m*d2theta2 + 4*I*d2theta2)

eq1 = sp.Eq(exp1, 0)
eq2 = sp.Eq(exp2, 0)
eq3 = sp.Eq(exp3, 0)

A, b = sp.linear_eq_to_matrix([eq1, eq2, eq3], [d2x, d2theta1, d2theta2])

sol = A.LUsolve(b)
d2x_expr, d2theta1_expr, d2theta2_expr = sol

# 1) Crie seis símbolos para usar no lambdify
x_sym, dx_sym, th1_sym, dth1_sym, th2_sym, dth2_sym = sp.symbols(
    'x_sym dx_sym th1_sym dth1_sym th2_sym dth2_sym'
)

subs_states = {
    x:        x_sym,     # onde for x(t), coloca-se x_sym
    dx:       dx_sym,    # onde for dx = x.diff(t), coloca-se dx_sym
    theta1:   th1_sym,   # onde for theta1(t), coloca-se th1_sym
    dtheta1:  dth1_sym,  # onde for dtheta1 = theta1.diff(t), -> dth1_sym
    theta2:   th2_sym,
    dtheta2:  dth2_sym
}

# 3) Aplique esse substituição nas expressões de aceleração
d2x_for_lam    = d2x_expr.subs(subs_states)
d2t1_for_lam   = d2theta1_expr.subs(subs_states)
d2t2_for_lam   = d2theta2_expr.subs(subs_states)

# 4) Dicionário de parâmetros numéricos
subs_params = {
    L:    0.3,
    m:    0.15,
    M:    0.3,
    g:    9.81,
    I:    (0.15 * 0.3**2) / 12,  
    alpha: 0.5**0.5,
    beta:  0.5**0.5
}

# 5) Faça a substituição de parâmetros primeiro
d2x_num    = sp.simplify(d2x_for_lam.subs(subs_params))
d2t1_num   = sp.simplify(d2t1_for_lam.subs(subs_params))
d2t2_num   = sp.simplify(d2t2_for_lam.subs(subs_params))

from sympy import lambdify

# 6) Gere as funções ddx_func, ddt1_func, ddt2_func que recebem 6 floats e devolvem um float
ddx_func = lambdify(
    (x_sym, dx_sym, th1_sym, dth1_sym, th2_sym, dth2_sym),
    d2x_num,
    'numpy'
)
ddt1_func = lambdify(
    (x_sym, dx_sym, th1_sym, dth1_sym, th2_sym, dth2_sym),
    d2t1_num,
    'numpy'
)
ddt2_func = lambdify(
    (x_sym, dx_sym, th1_sym, dth1_sym, th2_sym, dth2_sym),
    d2t2_num,
    'numpy'
)

import numpy as np
from scipy.integrate import solve_ivp

def ode_func(t, X):
    # X = [ x, dx, th1, dth1, th2, dth2 ]
    x_val, dx_val, th1_val, dth1_val, th2_val, dth2_val = X

    # Calcule as acelerações chamando as funções lambdificadas:
    ddx_val   = ddx_func(  x_val, dx_val,  th1_val, dth1_val,  th2_val, dth2_val )
    ddt1_val  = ddt1_func( x_val, dx_val,  th1_val, dth1_val,  th2_val, dth2_val )
    ddt2_val  = ddt2_func( x_val, dx_val,  th1_val, dth1_val,  th2_val, dth2_val )

    # Retorne a lista [dx, ddx, dth1, ddt1, dth2, ddt2]
    return [dx_val, ddx_val, dth1_val, ddt1_val, dth2_val, ddt2_val]

# 7) Condições iniciais (valores NumPy)
X0 = [0, 0, PI/4, 0, PI/2, 0]

t_final = 10.0
t_eval  = np.linspace(0.0, t_final, 1000)

sol = solve_ivp(
    ode_func,
    [0.0, t_final],
    X0,
    t_eval   = t_eval,
    method   = 'RK45',
    max_step = 0.01,
    atol     = 1e-8,
    rtol     = 1e-8
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(sol.t, sol.y[0], label='x(t) - bloco')
plt.plot(sol.t, sol.y[2], label='θ₁(t)')
plt.plot(sol.t, sol.y[4], label='θ₂(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posições / Ângulos')
plt.title('Simulação do sistema linearizado')
plt.legend()
plt.grid(True)
plt.show()
