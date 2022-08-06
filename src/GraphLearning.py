
from copy import deepcopy
import numpy as np
import scipy as sp
from scipy import optimize as op
import src.symmetries as sm
import src.GenerateGraph as gg
import matplotlib.pyplot as plt
import os

rng = np.random.RandomState()
seeded_rng = rng
os.chdir('..')
head_dir = os.getcwd()

textbooks = head_dir + "/textbook_networks/"

def learn(A, beta):
    '''
    Learning transform f(A).
    :param A: normalized transition matrix
    :param beta: accuracy parameter/inverse temperature
    :return: learned network f(A)
    '''
    A = normalize(A)
    inverse_argument = np.identity(len(A)) - np.exp(-beta)*A
    inverse = sp.linalg.inv(inverse_argument)
    return normalize((1-np.exp(-beta))*(A @ inverse))

def learn_to_undirected(A, beta, normalizer):
    out = learn(A, beta)
    out = unnormalize(out)
    return normalizer * out / np.sum(out)


def get_stationary(A, n_exp = 10000):
    '''
    Computes stationary distribution of A
    :param A: normalized transition matrix
    :param n_exp: base number of iterations
    :return: stationary distribution
    '''
    P = np.linalg.matrix_power(A, n_exp)
    P_next = np.dot(P, A)
    while not np.allclose(P, P_next):
        P_next = np.dot(P_next, A)
    return P_next[0]

def normalize(A):
    '''
    Normalizes an input adjacency matrix A into a transition matrix
    '''
    B = deepcopy(A)
    J = np.ones((len(A), len(A)))
    output = B / (B @ J)
    output[np.isnan(output)] = 0
    return output

def unnormalize(A):
    '''
    Constructs a weighted graph from a normalized transition matrix
    '''
    pi = get_stationary(A)
    return np.einsum('i, ij -> ij', pi, A)

def getNumEdges(A): #Assumes undirected input!
    return np.count_nonzero(A) / 2

def get_stationary_vec(A): #only applies for unnormalized weighted graph inputs
    output = np.sum(A, axis = 0) / (np.sum(A))
    return output


def KL_Divergence(U, V, weighted_net = None):
    U = normalize(U)
    V = normalize(V)
    if weighted_net is None:
        pi = get_stationary(U)
    else:
        pi = get_stationary_vec(weighted_net)
    combined = np.einsum('i, ij -> ij', pi, U)
    logged = np.log(V/U)
    logged[U == 0] = 0
    result = combined.T @ logged
    outcome = -np.trace(result)
    return outcome

def KL_score(A, beta, A_target = None):
    return KL_Divergence(A, learn(A, beta))


def uniformity_cost(P_0, A, beta):
    learned = learn(A, beta)
    terms = learned[P_0 > 0].flatten()
    diffs = np.subtract.outer(terms, terms)
    return np.sum(diffs * diffs)

def KL_score_external(A_input, beta, A_target, weighted_net = None):
    return KL_Divergence(A_target, learn(A_input, beta), weighted_net = weighted_net)


def get_pickleable_params(A, include_nonexistent = True, force_unique = False):
    comps, comp_maps, edge_labels, inv_labels = sm.unique_edges(A, force_unique = force_unique)
    if include_nonexistent:
        A_c = sm.get_complement_graph(A)
        comps_c, comp_maps_c, edge_labels_c, inv_labels_c = sm.unique_edges(A_c, force_unique = force_unique)
        return len(comps) + len(comps_c), comps, comps_c, inv_labels, inv_labels_c
    return len(comps), comps, None, inv_labels, None

def pickleable_cost_func(input, comps, comps_c, inv_labels, inv_labels_c, beta, A_target, include_nonexistent,
                         KL = True, weighted_net = None):
    B = np.zeros((len(A_target), len(A_target)))
    for i in range(len(comps)):
        for x in comps[i]:
            row, col = inv_labels[x]
            B[row][col], B[col][row] = input[i], input[i]
    if include_nonexistent:
        for i in range(len(comps_c)):
            for x in comps_c[i]:
                row, col = inv_labels_c[x]
                B[row][col], B[col][row] = input[len(comps) + i], input[len(comps) + i]
    if KL:
        return KL_score_external(B, beta, A_target, weighted_net = weighted_net)
    else:
        return uniformity_cost(A_target, B, beta)

def reduced_cost_func(input, comps, comps_c, inv_labels, inv_labels_c, beta, A_target, indices_c):
    B = np.zeros((len(A_target), len(A_target)))
    for i in range(len(comps)):
        for x in comps[i]:
            row, col = inv_labels[x]
            B[row][col], B[col][row] = input[i], input[i]
    for i in range(len(indices_c)):
        for x in comps_c[indices_c[i]]:
            row, col = inv_labels_c[x]
            B[row][col], B[col][row] = input[len(comps) + i], input[len(comps) + i]
    return KL_score_external(B, beta, A_target)

def one_param_cost_func(input, parameterized, beta, A_target):
    B = parameterized(input)
    return KL_score_external(B, beta, A_target)

def KL_modular_exemplar_general(input, N_tot, N_comms, beta):
    cc_bias, b_bias = input
    return KL_score_external(gg.modular_exemplar_general(N_tot, N_comms, cc_bias, b_bias), beta, gg.modular_exemplar_general(N_tot, N_comms, 1, 1))



def optimize_learnability(network0, weighted, symmInfo, parameterized, beta, include_nonexistent, KL = True, get_weights = False):
    numParams, comps, comps_c, inv_labels, inv_labels_c = symmInfo
    bounds = [(0, 1) for i in range(numParams)]
    outcome = op.dual_annealing(pickleable_cost_func, bounds=bounds,
        args=(comps, comps_c, inv_labels, inv_labels_c, beta, network0, include_nonexistent, KL, weighted),
                                                                accept = -40, maxiter = 1000, maxfun= 1.2e6)
    A = parameterized(outcome.x)
    score_original = KL_score(network0, beta)
    score = KL_score_external(A, beta, network0)
    if get_weights:
        return A, score_original, score, outcome.x
    return A, score_original, score

def optimize_one_param_learnability(network0, parameterized, beta, bound_max = 100):
    bounds = [(0, bound_max)]
    outcome = op.dual_annealing(one_param_cost_func, bounds = bounds,
                                args = (parameterized, beta, network0), accept = -10, maxiter=1000, maxfun = 1e6)
    A = parameterized(outcome.x[0])
    score_original = KL_score(network0, beta)
    score = KL_score_external(A, beta, network0)
    return A, outcome.x[0], score_original, score

if __name__ == '__main__':
    # load original networks
    networks_orig = np.load(textbooks + "cooc_mats.npy", allow_pickle=True)
    for i in range(len(networks_orig)):
        for j in range(len(networks_orig[i])):
            networks_orig[i][j][j] = 0
    for i in range(len(networks_orig)):
        networks_orig[i] /= np.sum(networks_orig[i])

    #load optimized networks
    networks = []
    scores = []
    betas = np.linspace(1e-3, .2, 15)
    for i in range(10):
        networks.append(np.load(textbooks + str(i) + "_opt_networks.npy", allow_pickle=True))
        scores.append(np.load(textbooks + str(i) + "_KL.npy", allow_pickle=True))
    for i in range(len(networks)):
        for j in range(len(betas)):
            networks[i][j] /= np.sum(networks[i][j])

    markers = ["o", "+", "*", "D", "x", "d", "^", "s", "v", ">"]
    colors = ["orange", "sienna", "limegreen", "deepskyblue", "steelblue", "purple", "lightseagreen", "darkgrey",
              "black", "red"]
    names = ["Treil", "Axler", "Edwards", "Lang", "Petersen", "Robbiano", "Bretscher", "Greub", "Hefferson", "Strang"]

    np.random.shuffle(colors)
    plt.figure(100, figsize=(6.5, 4.5))
    plt.rcParams.update({'font.size': 16})
    plt.xlabel(r'$\beta$')
    plt.ylabel('KL Divergence Ratio')
    plt.rcParams.update({'font.size': 16})
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # plt.ylim([0.25, 1.7])
        plt.scatter(betas, scores[i][:, 0] / scores[i][:, 1], s=30, alpha=.7, color=colors[i], marker=markers[i],
                    label=names[i])
        plt.plot(betas, scores[i][:, 0] / scores[i][:, 1], linewidth=.6, color=colors[i])
        plt.legend(frameon=False, prop={'size': 12}, labelspacing=.2, handletextpad=0, borderpad=0, loc='center left',
                   bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()













