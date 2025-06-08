from Matrizes import *

AB  =      A*B
A2B = (A**2)*B
A3B = (A**3)*B
A4B = (A**4)*B
A5B = (A**5)*B

Q = B.row_join(AB).row_join(A2B).row_join(A3B).row_join(A4B).row_join(A5B)

print(Q.rank())
# Printa 6 :D