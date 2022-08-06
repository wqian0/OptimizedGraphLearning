import src.symmetries as sm
import numpy as np
from src.GraphLearning import get_pickleable_params, pickleable_cost_func, normalize, KL_Divergence
from scipy import optimize as op
import matplotlib.pyplot as plt
import pickle as pk
import os
transient_data_path = os.getcwd()+'/transient_simulation_data'
rng = np.random.RandomState()
seeded_rng = rng

def transient_regime_simulation(unweighted, beta, n_steps = 1500):
    '''

    :param unweighted: unweighted/binary adjacency matrix
    :param beta: inverse temperature/accuracy param
    :param n_steps: number of transient steps
    :return: None
    '''
    numParams, comps, comps_c, inv_labels, inv_labels_c = get_pickleable_params(unweighted, include_nonexistent=False, force_unique= False)
    numParams, parameterized = sm.getSymReducedParams(unweighted, include_nonexistent=False, force_unique= False)
    bounds = [(0, 1) for j in range(numParams)]
    outcome = op.dual_annealing(pickleable_cost_func, bounds=bounds,
                                args=(
                                comps, comps_c, inv_labels, inv_labels_c, beta, unweighted, False, True, unweighted),
                                accept=-50, maxiter=2000, maxfun=300000)
    opt = parameterized(outcome.x)
    opt = normalize(opt)
    A = normalize(unweighted)
    n = len(A)
    A_opt = opt

    start = np.random.randint(n)
    all_data = [[],[],[]]

    plt.figure(1, figsize=(5.5, 4.5))
    plt.rcParams.update({'font.size': 16})
    for j in range(10):
        counts = (np.ones((n, n)) - np.identity(n))
        scrambled = (np.ones((n, n)) - np.identity(n))
        counts_op = (np.ones((n, n)) - np.identity(n))
        scrambled_opt = (np.ones((n, n)) - np.identity(n))
        belief_sc = np.zeros(n)
        belief_opt = np.zeros(n)
        MLE_score = []
        human_score = []
        opt_score = []
        start = np.random.randint(n)
        seq = [start]
        seq_opt = [start]
        next = 0
        next_op = 0
        for i in range(n_steps):
            next = rand_walk_step(A, next,counts)
            seq.append(next)
            update_scrambled_count(scrambled, beta, i+1, seq, belief_sc)

            next_op = rand_walk_step(A_opt,next_op,counts_op)
            seq_opt.append(next_op)
            update_scrambled_count(scrambled_opt, beta, i+1, seq_opt, belief_opt)
            MLE_score.append(KL_Divergence(A, normalize(counts)))
            human_score.append(KL_Divergence(A, normalize(scrambled)))
            opt_score.append(KL_Divergence(A, normalize(scrambled_opt)))

        if j == 0:
            plt.plot(human_score, color = 'blue', label = 'Human learning', linewidth = .5)
            plt.plot(MLE_score, color = 'orange', label  ='MLE',linewidth = .5)
            plt.plot(opt_score, color = 'red', label = 'Optimized human learning',linewidth = .5)
        else:
            plt.plot (human_score, color='blue',linewidth = .5)
            plt.plot(MLE_score, color='orange',linewidth = .5)
            plt.plot(opt_score, color='red',linewidth = .5)
        all_data[0].append(human_score)
        all_data[1].append(MLE_score)
        all_data[2].append(opt_score)
    plt.legend(frameon=False, prop={'size': 13.5}, labelspacing=.2, handletextpad=0, borderpad = 0,  loc = 3)
    plt.xlabel('Number of transitions observed')
    plt.ylabel('KL Divergence')
    plt.tight_layout()

def rand_walk_step(A, init, counts):
    candidates = np.nonzero(A[init])[0]
    next = seeded_rng.choice(candidates, p = A[init][candidates])
    counts[init][next] += 1
    return next

def update_scrambled_count(count, beta, t, seq, B):
    B *= np.exp(-beta)
    for i in range(len(count)):
        B[i] += seq[t-1] == i
    for i in range(len(count)):
        count[i][seq[-1]] +=  B[i]

if __name__ == '__main__':
    #load transient regime simulation data, computed via transient_regime_simulation()
    edwards = pk.load(open(transient_data_path+"/all_data.pk", "rb"))
    axler = pk.load(open(transient_data_path+"/all_data_axler.pk","rb"))
    modular = pk.load(open(transient_data_path+"/all_data_modular_10.pk", "rb"))
    all = [edwards, axler, modular]
    for j in range(3):
        plt.figure(j, figsize=(5.5, 4.5))
        plt.rcParams.update({'font.size': 16})
        for i in range(10):
            if i == 0:
                plt.plot(all[j][1][i], color='orange', label='Max. Likelihood Estimate', linewidth=.5)
                plt.plot(all[j][0][i], color='blue', label='Human learning', linewidth=.5)
                plt.plot(all[j][2][i], color='red', label='Optimized human learning', linewidth=.5)
            else:
                plt.plot(all[j][0][i],color='blue', linewidth = 0.5)
                plt.plot(all[j][1][i],color='orange', linewidth = 0.5)
                plt.plot(all[j][2][i],color='red', linewidth = 0.5)
        plt.legend(frameon=False, prop={'size': 13.5}, labelspacing=.5, handletextpad=.3, borderpad=.2, loc=0)
        plt.xlabel('Number of transitions observed')
        plt.ylabel('KL Divergence, ' + r'$D_{KL}(A||f(A_{in}))$')
        plt.tight_layout()
    plt.show()