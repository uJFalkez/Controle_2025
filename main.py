import sympy as sp
import copy

sin = sp.sin
cos = sp.cos
PI = sp.pi

SILENT = True

# Definição de Símbolos
x, theta1, theta2 = sp.symbols(r'x \tilde{\theta}_1 \tilde{\theta}_2')
dx, dtheta1, dtheta2 = sp.symbols(r'\dot{x} \dot{\tilde{\theta}}_1 \dot{\tilde{\theta}}_2')
d2x, d2theta1, d2theta2 = sp.symbols(r'\ddot{x} \ddot{\tilde{\theta}}_1 \ddot{\tilde{\theta}}_2')

g, L, m, M, F, Fp, T, Tp1, Tp2, I, alpha, beta = sp.symbols(r'g L m M \tilde{F} F_p \tilde{T} T_{p_1} T_{p_2} I \alpha \beta')

# Definição das equações do movimento
exp1 = 3*L*m*beta*d2theta1 + (4*m + 2*M)*d2x - 2*F - 2*Fp
exp2 = -6*g*L*m*alpha*theta1 + 5*L**2*m*d2theta1 - 2*L**2*m*alpha*d2theta2 + 6*L*m*beta*d2x + 4*I*d2theta1 - T - Tp1
exp3 = 2*g*L*m*theta2 - 2*L**2*m*alpha*d2theta1 + L**2*m*d2theta2 + 4*I*d2theta2 - Tp2

eq1 = sp.Eq(exp1, 0)
eq2 = sp.Eq(exp2, 0)
eq3 = sp.Eq(exp3, 0)

# Solução do sistema para d2x, d2theta1, d2theta2
A, b = sp.linear_eq_to_matrix([eq1.lhs, eq2.lhs, eq3.lhs], [d2x, d2theta1, d2theta2])

sol = A.LUsolve(-b)

d2x_expr, d2theta1_expr, d2theta2_expr = sol[0].simplify(), sol[1].simplify(), sol[2].simplify()

# Derivação do espaço de estados
dXdt = sp.Matrix([
    dx,             # d(x)/dt
    d2x_expr,       # d²(x)/dt²
    dtheta1,        # d(theta1)/dt
    d2theta1_expr,  # d²(theta1)/dt²
    dtheta2,        # d(theta2)/dt
    d2theta2_expr   # d²(theta2)/dt²
])

X_vetor = sp.Matrix([x, dx, theta1, dtheta1, theta2, dtheta2])
u_vetor = sp.Matrix([F, T])
p_vetor = sp.Matrix([Fp, Tp1, Tp2])

A = dXdt.jacobian(X_vetor).applyfunc(sp.simplify)
B = dXdt.jacobian(u_vetor).applyfunc(sp.simplify)
E = dXdt.jacobian(p_vetor).applyfunc(sp.simplify)

# Determinação dos valores numéricos
subs_dict = {
    L: 0.3,
    m: 0.06,
    M: 0.25,
    g: 9.81,
    I: 0.00045,
    alpha: 0.707,
    beta: 0.707
}

A_num = A.applyfunc(lambda exp: exp.subs(subs_dict))
B_num = B.applyfunc(lambda exp: exp.subs(subs_dict))
E_num = E.applyfunc(lambda exp: exp.subs(subs_dict))

# Derivação das funções de transferência
C_matrix = sp.Matrix([[1,0,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0,0,0,0,1,0]])

I_matrix = sp.eye(6)

s = sp.Symbol('s')

G_s_ = (C_matrix*((s*I_matrix-A_num).inv()*B_num)).applyfunc(sp.simplify)

G_s = G_s_.applyfunc(lambda e: e.evalf(n=3, chop=True))

G_x      = G_s[0], G_s[1]
G_theta1 = G_s[2], G_s[3]
G_theta2 = G_s[4], G_s[5]

# Determinação das raízes da função de transferência
positions = [x, theta1, theta2]
inputs = [F, T]
polos = {}
zeros = {}
for pos, line in zip(positions, (G_x, G_theta1, G_theta2)):
    temp_z = {}
    p = sp.Poly(sp.fraction(line[0])[1], s).all_roots()
    temp_p = [r.evalf() for r in p]
    for inp, exp in zip(inputs, line):
        num, _ = sp.fraction(exp)
        z = sp.Poly(num, s).all_roots()
        temp_z.update({inp:[r.evalf() for r in z]})
    zeros.update({pos:copy.deepcopy(temp_z)})
    polos.update({pos:copy.deepcopy(temp_p)})

if not SILENT:
    for pos, item in polos.items():
        print(f"{pos}:")
        for i, root in enumerate(item):
            print(f"Polo {i+1}: {root}")
        print()
        
    for pos, item in zeros.items():
        print(f"{pos}:")
        for inp, roots in item.items():
            print(f"{inp}:")
            for i, root in enumerate(roots):
                print(f"Zero {i+1}: {root}")
            print()
        print()
        

# Derivação da função de transferência com perturbações
G_sw_ = (C_matrix*((s*I_matrix-A_num).inv()*E_num)).applyfunc(sp.simplify)

G_sw = G_sw_.applyfunc(lambda e: e.evalf(n=3, chop=True))

G_xw      = G_sw[0], G_sw[1], G_sw[2]
G_theta1w = G_sw[3], G_sw[4], G_sw[5]
G_theta2w = G_sw[6], G_sw[7], G_sw[8]

# Determinação das raízes da função de transferência com perturbações
inputs_w = [Fp, Tp1, Tp2]
polos_w = {}
zeros_w = {}
for pos, line in zip(positions, (G_xw, G_theta1w, G_theta2w)):
    temp_z = {}
    p = sp.Poly(sp.fraction(line[0])[1], s).all_roots()
    temp_p = [r.evalf() for r in p]
    for inp, exp in zip(inputs_w, line):
        num, _ = sp.fraction(exp)
        z = sp.Poly(num, s).all_roots()
        temp_z.update({inp:[r.evalf() for r in z]})
    zeros_w.update({pos:copy.deepcopy(temp_z)})
    polos_w.update({pos:copy.deepcopy(temp_p)})

if not SILENT:
    for pos, item in polos_w.items():
        print(f"{pos}:")
        for i, root in enumerate(item):
            print(f"Polo {i+1}: {root}")
        print()
        
    for pos, item in zeros_w.items():
        print(f"{pos}:")
        for inp, roots in item.items():
            print(f"{inp}:")
            for i, root in enumerate(roots):
                print(f"Zero {i+1}: {root}")
            print()
        print()
        
# Diagramas de Bode
from scipy.signal import TransferFunction, bode
import matplotlib.pyplot as plt

class TF_Obj:
    """
    Essa classe inicializa as funções de transferência dependendo de qual pretende-se utilizar.
    Parameters:
        pos: A variável de estado respectiva da FT, deve ser "x", "theta1" ou "theta2"
        input_: A variável de entrada respectiva da FT, deve ser "F", "T", "Fp", "Tp1" ou "Tp2"
    """
    def __init__(self, pos: str, input_: str):
        pos_i = ("x", "theta1", "theta2").index(pos)
        self.pretty_pos = ("$x$", r"$\theta_1$", r"$\theta_2$")[pos_i]
        
        if "p" in input_:
            G_ = G_sw
            temp_i = 3
            input_i = ("Fp", "Tp1", "Tp2").index(input_)
            self.pretty_input = ("$F_p$", r"$T_{p_1}$", r"$T_{p_2}$")[input_i]
        else:
            G_ = G_s
            temp_i = 2
            input_i = ("F", "T").index(input_)
            self.pretty_input = ("$F$", r"$T$")[input_i]
            
        if pos_i == -1 or input_i == -1:
            raise ValueError
        
        index_G = pos_i*temp_i + input_i
        self.G = G_[index_G]

    def bodeInit(self):
        self.num, self.den = sp.fraction(self.G)
        num_coeffs = sp.Poly(self.num, s).all_coeffs()
        den_coeffs = sp.Poly(self.den, s).all_coeffs()

        num_coeffs = [float(c) for c in num_coeffs]
        den_coeffs = [float(c) for c in den_coeffs]

        sys = TransferFunction(num_coeffs, den_coeffs)
        self.w, self.mag, self.phase = bode(sys)

    def plot(self):
        plt.figure(figsize=(12, 6))

        # Magnitude
        plt.subplot(2, 1, 1)
        plt.semilogx(self.w, self.mag)
        plt.title(f'Diagrama de Bode (Relação {self.pretty_pos} vs {self.pretty_input})')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)

        # Fase
        plt.subplot(2, 1, 2)
        plt.semilogx(self.w, self.phase)
        plt.ylabel('Fase (graus)')
        plt.xlabel('Frequência (rad/s)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
tf = TF_Obj("theta2", "T")
tf.bodeInit()
tf.plot()