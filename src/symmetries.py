import igraph as ig
import networkx as nx
import numpy as np
import src.unionfind as uf

def automorphism_count(A, beta, A_target = None):
    IG = ig.Graph.Adjacency(A.tolist())
    return IG.count_automorphisms_vf2()

def get_automorphisms(A):
    IG = ig.Graph.Adjacency(A.tolist())
    return np.transpose(np.array(IG.get_automorphisms_vf2()))

def get_structurally_symmetric(A, force_unique = False):
    disj_set = uf.UnionFind()
    for i in range(len(A)):
        disj_set.add(i)
    if force_unique:
        return disj_set.components(), disj_set.component_mapping()
    autos = get_automorphisms(A)
    for i in range(len(A)):
        for j in range(len(autos[0])):
            disj_set.union(i, autos[i][j])
    return disj_set.components(), disj_set.component_mapping()

#input: symmetric adj matrix
def compute_line_graph(A):
    G_input = nx.from_numpy_matrix(A)
    return nx.to_numpy_matrix(nx.line_graph(G_input))

def unique_edges(A, A_target = None, force_unique = False):
    edge_labels, inv_labels, line_graph = compute_line_graph_details(A)
    components, comp_mappings = get_structurally_symmetric(line_graph, force_unique = force_unique)
    return components, comp_mappings, edge_labels, inv_labels

def compute_line_graph_details(A):
    edge_labels = dict()
    count = 0
    for r in range(len(A)-1):
        for c in range(r+1, len(A)):
            if A[r][c] == 1.0:
                edge_labels[(r,c)] = count
                count += 1
    inv_labels = {v: k for k, v in edge_labels.items()}
    keys = list(edge_labels.keys())
    edges = int(np.sum(A) / 2)
    for k in keys:
        r,c = k
        edge_labels[(c,r)] = edge_labels[(r,c)]
    output = np.zeros((edges, edges))
    for i in range(edges - 1):
        a, b = inv_labels[i]
        for j in range(i + 1, edges):
            c, d = inv_labels[j]
            if a == c or a == d or b == c or b == d:
                if not ((a == d and b == c) or (a == c and b == d)):
                    output[i][j], output[j][i] = 1, 1
    return edge_labels, inv_labels, output

def get_complement_graph(A):
    B = np.zeros((len(A), len(A)))
    B[A == 0] = 1
    B[A == 1] = 0
    for i in range(len(A)):
        B[i][i] = 0
    return B

def getSymReducedParams(A, include_nonexistent = True, force_unique = False):
    comps, comp_maps, edge_labels, inv_labels = unique_edges(A, 0, force_unique = force_unique)
    if include_nonexistent:
        A_c = get_complement_graph(A)
        comps_c, comp_maps_c, edge_labels_c, inv_labels_c = unique_edges(A_c, 0, force_unique = force_unique)
    def parameterized(input):
        B = np.zeros((len(A), len(A)))
        for i in range(len(comps)):
            for x in comps[i]:
                row, col = inv_labels[x]
                B[row][col], B[col][row] = input[i], input[i]
        if include_nonexistent:
            for i in range(len(comps_c)):
                for x in comps_c[i]:
                    row, col = inv_labels_c[x]
                    B[row][col], B[col][row] = input[len(comps) + i], input[len(comps)+i]
        return B
    if include_nonexistent:
        return len(comps) + len(comps_c), parameterized
    return len(comps), parameterized

