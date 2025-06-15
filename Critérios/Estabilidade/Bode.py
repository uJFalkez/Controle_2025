import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def Bode(A, B, C, D):
    out_labels = ['x', r'θ₁', r'θ₂']
    # rótulos das entradas
    in_labels  = ['F', 'T']

    # vetor de frequências (rad/s)
    w = np.logspace(-2, 2, 1000)

    # para cada entrada, monta uma figura com magnitude e fase
    for j in range(B.shape[1]):
        # extrai numerador e denominador da TF para essa entrada
        num, den = signal.ss2tf(A, B, C, D, input=j)
        
        # gera figura com 2 subplots (mag e phase)
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(f'Diagrama de Bode — entrada {in_labels[j]}')
        
        for i in range(C.shape[0]):
            b = num[i]  # numerador 1-D para o canal saída i ← entrada j
            # calcula bode
            w_out, mag, phase = signal.bode((b, den), w)
            # plota magnitude
            ax_mag.semilogx(w_out, mag, label=f'{out_labels[i]}')
            # plota fase
            ax_ph.semilogx(w_out, phase)
            ax_ph.set_yticks([-180, -90, 0, 90, 180])
            ax_ph.set_yticklabels(['-180°', '-90°', '0°', '90°', '180°'])
        
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.grid(True, which='both', ls=':')
        ax_mag.legend(loc='best', fontsize='small')
        
        ax_ph.set_xlabel('ω (rad/s)')
        ax_ph.set_ylabel('Fase')
        ax_ph.grid(True, which='both', ls=':')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()