import sympy as sp

alpha, beta, mu, kappa, eps, g, l, m, c, I_2, T_bar = sp.symbols("\\alpha \\beta \\mu \\kappa \\epsilon g l m c I_2 T_bar")

A1 = [0, 0, 1, 0, 0]
A2 = [0, 0, 0, 1, 0]

A31 = -(1/2)*(g**3*eps*kappa*mu)/alpha
A32 = -(1/3)*(g**3*l*m*eps*mu)/alpha
A35 = -(T_bar*c*g*mu)/alpha

A41 = (1/3)*(g**3*eps**2*mu)/alpha
A42 = ((2*g*l*m*kappa)*(-T_bar**2+g**2*(I_2+(7/6)*l**2*m)*mu))/(mu*alpha)
A45 = (2/3)*(T_bar*c*g*eps)/alpha

A51 = -(1/2)*(T_bar*g**2*eps*mu)/alpha
A52 = -(1/3)*(T_bar*g**2*l*m*eps)/alpha
A55 = c*g**2*((1/9)*eps**2-(I_2+(7/6)*l**2*m)*kappa)/alpha

B1 = [0, 0]
B2 = [0, 0]

B31 = (g*mu)/(T_bar*beta)
B32 = 1/beta

B41 = -((1/24)*g*eps*mu)/(T_bar*kappa*beta)
B42 = -((1/24)*eps)/(kappa*beta)

B51 = 1/beta
B52 = T_bar/(g*mu*beta) - 1/mu


A = sp.Matrix(
    [A1,
     A2,
     [A31, A32, 0, 0, A35],
     [A41, A42, 0, 0, A45],
     [A51, A52, 0, 0, A55]])

B = sp.Matrix(
    [B1,
     B2,
     [B31, B32],
     [B41, B42],
     [B51, B52]])

s = sp.symbols('s')

C = sp.Matrix([[1, 1, 0, 0, 1]])
D = sp.Matrix([[0, 0]])

I = sp.Matrix([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]])

G = C * (s*I - A).inv() * B + D

num, den = sp.fraction(G)

ordem1 = sp.degree(num, gen=s)
ordem2 = sp.degree(den, gen=s)

print(ordem1)
print(ordem2)



#G_simples = G.applyfunc(sp.simplify)

#with open("dump.txt",'w') as file:
#    file.write(sp.latex(G))
