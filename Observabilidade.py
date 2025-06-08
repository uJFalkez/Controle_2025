from Matrizes import *

AT = A.transpose()
CT = C.transpose()

ATCT  =      AT*CT
AT2CT = (AT**2)*CT
AT3CT = (AT**3)*CT
AT4CT = (AT**4)*CT
AT5CT = (AT**5)*CT

N = CT.row_join(ATCT).row_join(AT2CT).row_join(AT3CT).row_join(AT4CT).row_join(AT5CT)

print(N.rank())
# Printa 6 :D