from Valores_numéricos import *

#import Critérios.Estabilidade.Bode
#Critérios.Estabilidade.Bode.Bode(A, B, C, D)

#import Critérios.Estabilidade.Routh_Hurwitz
#Critérios.Estabilidade.Routh_Hurwitz.Routh(A)

#import Observadores.Trivial
#Observadores.Trivial.Observador_Trivial(A, B, C, K_LQR, POLOS)

#import Observadores.Ordem_Reduzida
#Observadores.Ordem_Reduzida.Observador_OR(A, B, C, K_LQR)

#import Simulações.Seguidor.Simu_pert
#Simulações.Seguidor.Simu_pert.Seguidor_Pert(A, B, C, L_or, K_LQR

#import Simulações.LQR.Simu
#Simulações.LQR.Simu.Controlador_LQR(A, B)

#import Simulações.LQR.Simu_pert
#Simulações.LQR.Simu_pert.Controlador_LQR_Pert(A, B)

#import Simulações.Seguidor.Simu_pert
#Simulações.Seguidor.Simu_pert.Seguidor_Pert(A, B, C, E, L_or, K_LQR)

import Simulações.Seguidor.Simu_pert_rej
Simulações.Seguidor.Simu_pert_rej.Seguidor_Pert_Rej(A, B, C, E, L_or, K_LQR)