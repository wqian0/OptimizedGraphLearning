import numpy as np
import bct
import networkx as nx
def all_core_periphery_avged(networks_orig, networks_opt):
    output = np.zeros((15, 4))
    output_std = np.zeros((15, 4))
    classified = [[[], [], [], []] for _ in range(15)]
    for j in range(10):
        classifications, per_comm_assignments, _, _ = core_periphery_analysis(networks_orig[j])
        for i in range(15):
            classified_vals, _ = classify_vals(classifications, networks_orig[j], networks_opt[j][i], per_comm_assignments)
            for k in range(len(classified_vals)):
                classified[i][k].extend(classified_vals[k])
    for i in range(15):
        for k in range(4):
            output[i][k] = np.mean(classified[i][k])
            output_std[i][k] = np.std(classified[i][k])
    return output, output_std, classified

def all_core_periphery(networks_orig, networks_opt):
    output = np.zeros((10, 15, 4))
    output_std = np.zeros((10, 15, 4))
    for j in range(10):
        classifications, per_comm_assignments, _, _ = core_periphery_analysis(networks_orig[j])
        for i in range(15):
            classified_vals, _ = classify_vals(classifications, networks_orig[j], networks_opt[j][i], per_comm_assignments)
            for k in range(len(classified_vals)):
                output[j][i][k] = np.mean(classified_vals[k])
                output_std[j][i][k] = np.std(classified_vals[k])
    return output, output_std

def core_periphery_analysis(network0, save_out = 'axler_periphery.txt'):

    network0 /= np.sum(network0)
    C, Q_core = bct.core_periphery_dir(network0)
    per_nodes = []
    for i in range(len(C)):
        if C[i] == 0:
            per_nodes.append(i)
    G = nx.from_numpy_matrix(network0)
    G_per = G.subgraph(per_nodes)
    np.savetxt(save_out, nx.to_numpy_matrix(G_per))
    per_network = np.array(nx.to_numpy_matrix(G_per))
    M_per, Q_comm_per = bct.community_louvain(per_network)
    per_comm_assignments = {}
    for i in range(len(per_nodes)):
        per_comm_assignments[per_nodes[i]] = M_per[i]
    classifications = [[], [], []] # index 0 means periphery-periphery edge, 1 means periphery-core, 2 means core-core
    for i in range(len(network0) - 1):
        for j in range(i+1, len(network0)):
            if network0[i][j] > 0:
                classifications[C[i] + C[j]].append((i, j))
    return classifications, per_comm_assignments, G_per, M_per

def get_edge_values(A_original, A):
    edge_factors = {}
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            if A_original[i][j] > 0:
                edge_factors[(i, j)] = A[i][j]/A_original[i][j]
    return edge_factors

def classify_vals(classifications, network0, network_opt, per_comm_assignments):
    classified_vals = [[], [], [],
                       []]  # periphery-periphery intramod, periphery-periphery intermod, periphery-core, core-core
    classified_edges = [[], [], [], []]
    edge_factors = get_edge_values(network0, network_opt)
    for j in range(len(classifications[0])):
        e0, e1 = classifications[0][j]
        if per_comm_assignments[e0] == per_comm_assignments[e1]:
            classified_vals[0].append(edge_factors[classifications[0][j]])
            classified_edges[0].append(classifications[0][j])
        else:
            classified_vals[1].append(edge_factors[classifications[0][j]])
            classified_edges[1].append(classifications[0][j])
    for i in range(1, 3):
        for j in range(len(classifications[i])):
            classified_vals[i + 1].append(edge_factors[classifications[i][j]])
            classified_edges[i + 1].append(classifications[i][j])
    return classified_vals, classified_edges