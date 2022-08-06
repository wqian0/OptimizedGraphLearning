import numpy as np
from copy import deepcopy
import networkx as nx
import scipy as sp
rng = np.random.RandomState()
seeded_rng = rng

'''
Functions for producing generative model graphs
'''
def get_random_modular(n, modules, edges, p, getCommInfo=False):
    pairings = {}
    assignments = np.zeros(n, dtype = int)
    cross_module_edges = []
    for i in range(modules):
        pairings[i] = []
    A = np.zeros((n,n))
    for i in range(n):
        randomModule = seeded_rng.randint(0, modules)
        pairings[randomModule].append(i)
        assignments[i] = randomModule
    def add_modular_edge():
        randomComm = seeded_rng.randint(0, modules)
        while len(pairings[randomComm]) < 2:
            randomComm = seeded_rng.randint(0, modules)
        selection = seeded_rng.choice(pairings[randomComm], 2, replace=False)
        while A[selection[0], selection[1]] != 0:
            randomComm = seeded_rng.randint(0, modules)
            while len(pairings[randomComm]) < 2:
                randomComm = seeded_rng.randint(0, modules)
            selection = seeded_rng.choice(pairings[randomComm], 2, replace=False)
        A[selection[0], selection[1]] += 1
        A[selection[1], selection[0]] += 1

    def add_between_edge():
        randEdge = seeded_rng.choice(n, 2, replace=False)
        while A[randEdge[0], randEdge[1]] != 0 or assignments[randEdge[0]] == assignments[randEdge[1]]:
            randEdge = seeded_rng.choice(n, 2, replace=False)
        A[randEdge[0], randEdge[1]] += 1
        A[randEdge[1], randEdge[0]] += 1
        cross_module_edges.append(randEdge)
    inModuleEdges = int(round(edges * p))
    betweenEdges = edges - inModuleEdges
    # betweenEdges = edges - inModuleEdges - modules + 1
    # if betweenEdges < 0:
    #     print("NEGATIVE")
    for i in range(inModuleEdges):
        add_modular_edge()
    for i in range(betweenEdges):
        add_between_edge()
    def parameterized(cc_weight):
        B = deepcopy(A)
        for e in cross_module_edges:
            B[e[0], e[1]], B[e[1], e[0]] = cc_weight, cc_weight
        return B
    if getCommInfo:
        return A, parameterized, pairings, assignments
    else:
        return A, parameterized

def get_hierarchical_modular(n, modules, edges, p, alpha, getCommInfo=False):
    pairings = {}
    assignments = np.zeros(n, dtype = int)
    cross_module_edges = []
    weights = np.array([(1 + i) ** -alpha for i in range(n)])
    dists = []
    module_dist = np.zeros(modules)
    for i in range(modules):
        pairings[i] = []
    A = np.zeros((n,n))
    for i in range(n):
        randomModule = seeded_rng.randint(0, modules)
        pairings[randomModule].append(i)
        assignments[i] = randomModule
    for j in range(modules):
        dist = np.array([weights[i] for i in pairings[j]])
        module_dist[j] = np.sum(dist)
        dist /= np.sum(dist)
        dists.append(dist)
    module_dist /= np.sum(module_dist)
    # nodesPerMod = n // modules
    # for i in range(modules):
    #     for j in range(nodesPerMod):
    #         pairings[i].append(nodesPerMod * i + j)
    #         assignments[nodesPerMod *i + j] = i
    # for i in range(modules - 1):
    #     if len(pairings[i]) < 3 or len(pairings[i+1]) < 3:
    #         return None, None
    #     e0, e1 = seeded_rng.choice(pairings[i], 1), seeded_rng.choice(pairings[i+1], 1)
    #     A[e0, e1], A[e1, e0] = 1, 1
    #     cross_module_edges.append((e0, e1))
    def add_modular_edge():
        randomComm = seeded_rng.choice(modules, p = module_dist)
        while len(pairings[randomComm]) < 2:
            randomComm = seeded_rng.choice(modules, p = module_dist)
        selection = seeded_rng.choice(pairings[randomComm], 2, replace=False, p = dists[randomComm])
        while A[selection[0], selection[1]] != 0:
            randomComm = seeded_rng.choice(modules, p = module_dist)
            while len(pairings[randomComm]) < 2:
                randomComm = seeded_rng.choice(modules, p = module_dist)
            selection = seeded_rng.choice(pairings[randomComm], 2, replace=False, p = dists[randomComm])
        A[selection[0], selection[1]] += 1
        A[selection[1], selection[0]] += 1

    def add_between_edge():
        randomComm, randomComm2, e0, e1 = 0, 0, 0, 0
        while randomComm == randomComm2 or A[e0, e1] != 0:
            randomComm, randomComm2 = seeded_rng.choice(modules, p = module_dist), seeded_rng.choice(modules, p = module_dist)
            e0 = seeded_rng.choice(pairings[randomComm], 1, replace=False, p=dists[randomComm])
            e1 = seeded_rng.choice(pairings[randomComm2], 1, replace=False, p=dists[randomComm2])
        A[e0, e1] += 1
        A[e1, e0] += 1
        cross_module_edges.append((e0, e1))
    inModuleEdges = int(round(edges * p))
    betweenEdges = edges - inModuleEdges
    # betweenEdges = edges - inModuleEdges - modules + 1
    # if betweenEdges < 0:
    #     print("NEGATIVE")
    for i in range(inModuleEdges):
        add_modular_edge()
    for i in range(betweenEdges):
        add_between_edge()
    def parameterized(cc_weight):
        B = deepcopy(A)
        for e in cross_module_edges:
            B[e[0], e[1]], B[e[1], e[0]] = cc_weight, cc_weight
        return B
    if getCommInfo:
        return A, parameterized, pairings, assignments
    else:
        return A, parameterized

def create_network(N, edges):
    adjMatrix = np.zeros((N, N), dtype = float)
    for i in range(edges):
        randEdge = seeded_rng.choice(N, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = seeded_rng.choice(N, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1.0
    return adjMatrix

def create_undirected_network(N, edges):
    adjMatrix = np.zeros((N, N), dtype = float)
    perm = np.arange(N)
    for i in range(N - 1):
        adjMatrix[perm[i]][perm[i+1]] = 1.0
        adjMatrix[perm[i+1]][perm[i]] = 1.0
    for i in range(edges - N + 1):
        randEdge = seeded_rng.choice(N, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = seeded_rng.choice(N, 2, replace=False)
            #print("STUCK")
        adjMatrix[randEdge[0]][randEdge[1]] = 1.0
        adjMatrix[randEdge[1]][randEdge[0]] = 1.0
    return adjMatrix

def create_modular_toy(edges, modular_edges):
    adjMatrix = np.zeros((15, 15))
    module_1, module_2, module_3 = np.arange(5), np.arange(5, 10), np.arange(10, 15)
    modules = []
    modules.append(module_1), modules.append(module_2), modules.append(module_3)
    def add_modular_edge():
        randomComm = seeded_rng.randint(0, 3)
        randEdge = seeded_rng.choice(modules[randomComm], 2, replace = False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0 or adjMatrix[randEdge[1]][randEdge[0]] != 0:
            randomComm = seeded_rng.randint(0, 3)
            randEdge = seeded_rng.choice(modules[randomComm], 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
        adjMatrix[randEdge[1]][randEdge[0]] += 1

    def add_cross_edge(): #adds edge outside modules
        randEdge = seeded_rng.choice(15, 2, replace = False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0 or adjMatrix[randEdge[1]][randEdge[0]] != 0 or \
        (randEdge[0] in module_1 and randEdge[1] in module_1) \
                or (randEdge[0] in module_2 and randEdge[1] in module_2) or \
                (randEdge[0] in module_3 and randEdge[1] in module_3):
            randEdge = seeded_rng.choice(15, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
        adjMatrix[randEdge[1]][randEdge[0]] += 1
    for i in range(modular_edges):
        add_modular_edge()
    for i in range(edges - modular_edges):
        add_cross_edge()
    return adjMatrix

def modular_graph_exemplar():
    result = np.zeros((15, 15))
    for i in range(5):
        for j in range(5):
            result[i][j] += 1.0
    for i in range(5, 10):
        for j in range(5, 10):
            result[i][j] += 1.0
    for i in range(10, 15):
        for j in range(10, 15):
            result[i][j] += 1.0
    for i in range(15):
        result[i][i] = 0

    result[0][4], result[4][0] = 0, 0
    result[0][14], result[14][0] = 1, 1

    result[5][9], result[9][5] = 0, 0
    result[4][5], result[5][4] = 1, 1

    result[9][10], result[10][9] = 1, 1
    result[10][14], result[14][10] =0, 0
    return result

def biased_modular(cross_cluster_bias, boundary_bias):
    result = modular_graph_exemplar()
    result[0][14], result[14][0] = cross_cluster_bias, cross_cluster_bias
    result[4][5], result[5][4] = cross_cluster_bias, cross_cluster_bias
    result[9][10], result[10][9] = cross_cluster_bias, cross_cluster_bias
    for i in [0, 4]:
        for j in range(1, 4):
            result[i][j], result[j][i] = boundary_bias, boundary_bias

    for i in [5, 9]:
        for j in range(6, 9):
            result[i][j], result[j][i] = boundary_bias, boundary_bias

    for i in [10, 14]:
        for j in range(11, 14):
            result[i][j], result[j][i] = boundary_bias, boundary_bias
    return result

def get_laplacian(A):
    result = -deepcopy(A)
    for i in range(len(result[0])):
        result[i][i] += np.sum(A[i])
    return result

def isConnected(A, returnInfo = False):
    L = get_laplacian(A)
    evals, v = sp.linalg.eigh(L, eigvals=(1, len(A) - 1))
    if returnInfo:
        return evals[0] > 1e-10, evals
    return evals[0] > 1e-10

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def flipEdge(A, ensureConnected = True):
    B = deepcopy(A)
    edges = []
    empty = []
    for r in range(len(A) - 1):
        for c in range(r + 1, len(A)):
            if B[r][c] != 0.0 and B[r][c] != 0:
                edges.append((r, c))
            else:
                empty.append((r, c))
    rE = seeded_rng.randint(len(edges))
    rA = seeded_rng .randint(len(empty))
    B[edges[rE][0]][edges[rE][1]] = 0.0
    B[edges[rE][1]][edges[rE][0]] = 0.0
    B[empty[rA][0]][empty[rA][1]] = 1.0
    B[empty[rA][1]][empty[rA][0]] = 1.0
    if ensureConnected:
        if not isConnected(B):
            B = flipEdge(A)
        else:
            return B
    return B

def find_edgepair(A, e0, e1):
    newEdges = []
    if A[e0[0]][e1[0]] == 0.0 and A[e0[1]][e1[1]] == 0.0 and e0[0] != e1[0] and e0[1] != e1[1]:
        newEdges.append([(e0[0], e1[0]), (e0[1], e1[1])])
    if A[e0[1]][e1[0]] == 0.0 and A[e0[0]][e1[1]] == 0.0 and e0[1] != e1[0] and e0[0] != e1[1]:
        newEdges.append([(e0[1], e1[0]), (e0[0], e1[1])])
    if len(newEdges) == 0:
        return None
    else:
        return newEdges[seeded_rng.choice(len(newEdges))]

def rewire_regular(A, ensureConnected = True):
    B = deepcopy(A)
    newEdges = None
    dels, adj0, adj1 = 0,0,0
    while newEdges == None or adj0 == adj1:
        dels = seeded_rng.choice(len(B), 2, replace=False)
        adj0 = seeded_rng.choice(np.nonzero(B[dels[0]])[0])
        adj1 = seeded_rng.choice(np.nonzero(B[dels[1]])[0])
        newEdges = find_edgepair(B, (dels[0], adj0), (dels[1], adj1))

    B[dels[0]][adj0] = 0
    B[adj0][dels[0]] = 0
    B[dels[1]][adj1] = 0
    B[adj1][dels[1]] = 0
    B[newEdges[0][0]][newEdges[0][1]] = 1
    B[newEdges[0][1]][newEdges[0][0]] = 1
    B[newEdges[1][0]][newEdges[1][1]] = 1
    B[newEdges[1][1]][newEdges[1][0]] = 1
    if ensureConnected:
        if not isConnected(B):
            B = rewire_regular(A, ensureConnected = True)
        else:
            return B
    return B

def get_regular_graph(N, d):
    return np.array(nx.to_numpy_matrix(nx.random_regular_graph(d, N)))

def get_lattice_graph(dim):
    return np.array(nx.to_numpy_matrix(nx.grid_graph(dim, periodic = True)))

def compose(f, n):
    def fn(x):
        for _ in range(n):
            x = f(x)
        return x
    return fn

def transitivity_score(A, beta):
    G = nx.from_numpy_matrix(A)
    return nx.transitivity(G)

def symmetry_toy():
    A = np.zeros((15, 15))
    for i in range(5):
        for j in range(i + 1, 6):
            A[i][j], A[j][i] = 1, 1
    for i in range(8, 15):
        A[i][6], A[6][i] = 1, 1
        A[i][7], A[7][i] = 1, 1
    A[5][6], A[6][5] = 1, 1
    return A

def optimize(A, beta, iterations, scoreFunc, flipFunc, minimize = True, A_target = None, numParams = False):
    bestVal = float('inf')
    curr = deepcopy(A)
    best = deepcopy(A)
    factor = 1
    if not minimize:
        factor = -1
        bestVal = -float('inf')
    for i in range(iterations):
        if numParams:
            paramCount, parameterized = scoreFunc(curr)
            score = paramCount
        else:
            score = scoreFunc(curr, beta, A_target)
        currScore = factor * score
        if currScore <= bestVal and isConnected(curr):
            bestVal = currScore
            best = deepcopy(curr)
        curr = compose(flipFunc, seeded_rng.randint(len(A)))(best)
        count = 0
        while not isConnected(curr):
            curr = compose(flipFunc, seeded_rng.randint(len(A)))(best)
            count += 1
            if count > 30:
                return bestVal, best
        print(str(i)+"\t"+str(currScore)+"\t"+str(bestVal))
    return bestVal,  best


def modular_exemplar_general(N_tot, N_comms, cc_bias, b_bias):
    A = np.zeros((N_tot, N_tot))
    N_in = N_tot // N_comms
    b = [] #boundary nodes
    for i in range(N_comms):
        for j in range(i * N_in, (i + 1) * N_in - 1):
            for k in range(j + 1, (i + 1) * N_in):
                A[j][k], A[k][j] = 1.0, 1.0
        A[i * N_in][(i + 1) * N_in - 1], A[(i + 1) * N_in - 1][i * N_in] = 0, 0

    for i in range(N_comms):
        b.append(i * N_in)
        b.append((i + 1) * N_in - 1)
        for j in [i * N_in, (i + 1) * N_in - 1]:
            for k in range(i * N_in + 1, (i + 1) * N_in - 1):
                A[j][k], A[k][j] = b_bias, b_bias
    for i in range(1, len(b) // 2):
        A[b[2*i - 1]][b[2*i]], A[b[2*i]][b[2*i - 1]] = cc_bias, cc_bias
    A[b[0]][b[len(b) - 1]], A[b[len(b) - 1]][b[0]] = cc_bias, cc_bias
    return A

def stoch_block_parameterized(blocks, p_cc, p_in):
    pMatrix = np.zeros((len(blocks), len(blocks)))
    comms = {}
    k = 0
    for i in range(len(blocks)):
        for j in range(blocks[i]):
            comms[k] = i
            k += 1
    for i in range(len(pMatrix)):
        for j in range(len(pMatrix)):
            if i == j:
                pMatrix[i][j] = p_in
            else:
                pMatrix[i][j] = p_cc
    G = nx.generators.stochastic_block_model(blocks, pMatrix)
    A = np.array(nx.to_numpy_matrix(G))
    def parameterized(cc_weight):
        B = deepcopy(A)
        for i in range(len(A)):
            for j in range(len(A)):
                if comms[i] is not comms[j] and A[i][j] != 0:
                    B[i][j], B[j][i] = cc_weight, cc_weight
        return B
    return A, parameterized

def get_degrees(A):
    return np.array([sum(A[i]) for i in range(len(A))])

def small_world_parameterized(N_tot, k, p, getLatticeInfo = False):
    A_0 = np.array(nx.to_numpy_matrix(nx.generators.watts_strogatz_graph(N_tot, k, 0)))
    A = np.array(nx.to_numpy_matrix(nx.generators.watts_strogatz_graph(N_tot, k, p)))
    non_lattice_edges = []
    for i in range(N_tot - 1):
        for j in range(i + 1, N_tot):
            if A[i][j] != A_0[i][j] and A[i][j] == 1:
                non_lattice_edges.append((i, j))
    def parameterized(nl_weight):
        B = deepcopy(A)
        for e in non_lattice_edges:
            B[e[0]][e[1]], B[e[1]][e[0]] = nl_weight, nl_weight
        return B
    if getLatticeInfo:
        return A, parameterized, non_lattice_edges
    return A, parameterized

def sierpinski_generator(n, p): #generates a generalized Sierpinski graph with n hierarchical levels and p communities
    A = np.zeros((p ** n, p ** n))
    for k in range(n):
        for i in range(p):
            for j in range(p):
                for m in range(p ** (n - k - 1)):
                    if i != j:
                        e0 = m * (p ** (k + 1)) + i * (p ** k)  + j * ((p ** k) - 1) // (p - 1)
                        e1 = m * (p ** (k + 1)) + j * (p ** k) + i * ((p ** k) - 1) // (p - 1)
                        A[e0][e1] = 1
    return A

def regularized_sierpinski(n, p):
    A = np.zeros((p ** n + p ** (n-1), p ** n + p ** (n-1)))
    A_nonreg = sierpinski_generator(n, p)
    A_nonreg2 = sierpinski_generator(n-1, p)
    for i in range(len(A_nonreg)):
        for j in range(len(A_nonreg)):
            A[i][j] = A_nonreg[i][j]
    for i in range(len(A_nonreg2)):
        for j in range(len(A_nonreg2)):
            A[len(A_nonreg) + i][len(A_nonreg) + j] = A_nonreg2[i][j]
    for i in range(p):
        e0 = len(A_nonreg) + i * (p ** (n-1) - 1) // (p - 1)
        e1 = i * (p ** (n) - 1 )// (p - 1)
        A[e0][e1], A[e1][e0] = 1, 1
    return A

def printArrayToFile(A, file):
    for r in range(len(A)):
        file.write(str(A[r]) + "\t")
    file.write("\n")