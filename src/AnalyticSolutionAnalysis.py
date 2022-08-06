import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import src.GenerateGraph as gg
from src.GraphLearning import get_pickleable_params, pickleable_cost_func, normalize, KL_score_external
import src.symmetries as sm
import matplotlib as mpl
from scipy import optimize as op
def get_optimal_analytic(A_target, beta, get_cond_number = False):
    I = np.identity(len(A_target))
    inv_argument = I*(1-np.exp(-beta)) + np.exp(-beta)*A_target
    cond_number = np.linalg.cond(inv_argument, p='fro')
    inv = sp.linalg.pinv(inv_argument)
    if get_cond_number:
        return inv @ A_target, cond_number
    else:
        return inv @ A_target

if __name__ == '__main__':
    betas = np.linspace(1e-3, 1, 100)
    cc_direct, b_direct, d_direct = [], [], []
    cc_weights, b_weights = [], []
    cond_numbers = []
    network0 = gg.modular_graph_exemplar()
    A = normalize(network0)
    n = len(A)
    for i in range(len(betas)):
        A = normalize
        A_inv, cond_number = get_optimal_analytic(A, betas[i],get_cond_number=True)
        cond_numbers.append(cond_number)

        cc_direct.append(A_inv[0][14])
        b_direct.append(A_inv[0][1])
        d_direct.append(A_inv[1][2])

    plt.axis('off')
    plt.figure(4, figsize=(5.5, 4.5), dpi=1000)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel(r'$\beta$')
    plt.ylabel('Optimal ' +r'$A^*$'+ ' Value')
    plt.yscale('symlog')
    plt.plot(betas, cc_direct, color = 'orange', label = r'$A^*_{cc}$')
    plt.plot(betas, b_direct, color = 'forestgreen', label = r'$A^*_{b}$')
    plt.plot(betas, d_direct, color = 'grey', label = r'$A^*_{d}$')


    plt.figure()
    plt.figure(5, figsize=(5.5, 4.5), dpi=1000)
    plt.rcParams.update({'font.size': 15})
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$[(1 - e^{-\beta})I + e^{-\beta}A]$'+" Cond. Number")
    plt.yscale('log')
    plt.plot(betas, cond_numbers)
    plt.tight_layout()
    plt.savefig('direct_cond_numbers_mod.pdf')
    plt.show()