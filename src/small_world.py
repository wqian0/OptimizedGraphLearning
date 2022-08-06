'''
Source: https://github.com/sartorg/robinmax/blob/master/robinmax_graph_generator.py
'''
from __future__ import division

import numpy        as     np
from scipy.linalg import circulant


def _distance_matrix(L):
    Dmax = L // 2

    D = list(range(Dmax + 1))
    D += list(D[-2 + (L % 2):0:-1])

    return circulant(D) / Dmax


def _pd(d, p0, beta):
    return beta * p0 + (d <= p0) * (1 - beta)


def watts_strogatz(L, p0, beta, directed=False, rngseed=1):
    """
    Watts-Strogatz model of a small-world network

    This generates the full adjacency matrix, which is not a good way to store
    things if the network is sparse.

    Parameters
    ----------
    L        : int
               Number of nodes.

    p0       : float
               Edge density. If K is the average degree then p0 = K/(L-1).
               For directed networks "degree" means out- or in-degree.

    beta     : float
               "Rewiring probability."

    directed : bool
               Whether the network is directed or undirected.

    rngseed  : int
               Seed for the random number generator.

    Returns
    -------
    A        : (L, L) array
               Adjacency matrix of a WS (potentially) small-world network.

    """
    rng = np.random.RandomState(rngseed)

    d = _distance_matrix(L)
    p = _pd(d, p0, beta)

    if directed:
        A = 1 * (rng.random_sample(p.shape) < p)
        np.fill_diagonal(A, 0)
    else:
        upper = np.triu_indices(L, 1)

        A = np.zeros_like(p, dtype=int)
        A[upper] = 1 * (rng.rand(len(upper[0])) < p[upper])
        A.T[upper] = A[upper]

    return A