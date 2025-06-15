import sympy as sp

def Routh(A):
    s = sp.Symbol("s")
    I = sp.eye(6)
    
    print(sp.latex((s*I - A).det()))
    # Com isso, podemos usar uma ferramenta para construção de tabelas de Routh
    # Verifica-se que é estável: apenas a primeira linha tem valores não-nulos, portanto, o sistema é estável
    # Isso é de se esperar por não haver nenhum amortecimento.