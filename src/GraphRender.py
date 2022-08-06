
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import networkx as nx
import src.GenerateGraph as gg
import src.GraphLearning as gl
import src.symmetries as sm

def get_lattice_layout():
    '''
    hardcoded geometric layout of the 15-node lattice graph exemplar
    '''

    graph_pos = {}
    rad_in = .25
    rad_out = 1.2
    left = (-rad_in, 0)
    right = (rad_in, 0)
    top = (0, 2 * rad_in)
    centers = [(np.cos(np.pi / 2 + i  * 2 * np.pi / 5) * rad_out , np.sin(np.pi / 2 + i  * 2 * np.pi / 5) * rad_out) for i in range(5)]
    for i in range(5):
        for j in range(3):
            if j == 0:
                graph_pos[3 *i + j] = (centers[i][0] + top[0], centers[i][1] + top[1])
            if j == 1:
                graph_pos[3 * i + j] = (centers[i][0] + left[0], centers[i][1] + left[1])
            if j == 2:
                graph_pos[3 * i + j] = (centers[i][0] + right[0], centers[i][1] + right[1])
    return graph_pos

class MidpointNormalize(colors.Normalize):
	"""
	Source: http://chris35wills.github.io/matplotlib_diverging_colorbar/
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def get_modular_layout():
    '''
    hardcoded geometric layout of the 15-node modular graph examplar
    '''

    graph_pos = {}
    top = (0, .8)
    left = (-.5, 0)
    right = (.5, 0)
    rad = .25
    for i in range(5):
        x_off = -np.sin(2 * np.pi / 5 * (i +.5)) * rad
        y_off = -np.cos(2 * np.pi / 5  * (i + .5)) * rad
        graph_pos[i] = (top[0] + x_off, top[1] + y_off)
    for i in range(5):
        x_off = -np.sin(2 * np.pi / 5  * (i + 2)) * rad
        y_off = -np.cos(2 * np.pi / 5  * (i + 2)) * rad
        graph_pos[5 + i] = (right[0] + x_off, right[1] + y_off)
    for i in range(5):
        x_off = np.sin(2 * np.pi / 5  * (i +1.5)) * rad
        y_off = np.cos(2 * np.pi / 5  * (i + 1.5)) * rad
        graph_pos[10 + i] = (left[0] + x_off, left[1] + y_off)
    return graph_pos

def heatmap_render_modular(param_cap = 2, param_res = 500):
    beta_range = np.linspace(1e-3, param_cap, param_res)
    lambda_cc_range = np.linspace(1e-3, param_cap, param_res)
    lambda_b_range = np.linspace(1e-3, param_cap, param_res)
    results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))


    network0 = gg.modular_graph_examplar()

    for i in range(param_res):
        beta = beta_range[i]
        for j in range(len(lambda_b_range)):
            A_init = gg.biased_modular(lambda_cc_range[i], lambda_b_range[j])
            A_learned = gl.learn(A_init, beta)
            score_ext = gl.KL_score_external(A_init, beta, network0)
            score_baseline = gl.KL_score(network0, beta)
            results[i][j] = score_ext/score_baseline
    plt.figure(5)
    plt.rcParams.update({'font.size': 14})
    cax = plt.imshow(results, cmap = 'RdBu',extent=[.01, 2, .01, 2], origin='lower', vmax = 1.1, vmin = .9, aspect = 1, norm = mn.MidpointNormalize(midpoint=1))
    plt.title("KL Divergence Ratio", size=18)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel(r"$\lambda _{b}$", size=20)
    plt.ylabel(r"$\lambda _{cc}$", size=20)
    cbar = plt.colorbar(cax, ticks = [.9, .95, 1.0, 1.05, 1.1])
    cbar.ax.set_yticklabels(['<.9','.95','1.0','1.05', '> 1.1'])
    plt.tight_layout()
    plt.show()
def heatmap_render(betaCap, lam1Cap, lam2Cap): #compute and plot a heatmap sweep of edge weighting parameters
    beta_range = np.linspace(1e-3, betaCap, 500)
    lambda_cc_range = np.linspace(1e-3, lam1Cap, 500)
    lambda_b_range = np.linspace(1e-3, lam2Cap, 500)
    results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))

    network0 = gg.get_lattice_graph([3,5])
    numParams, parameterized = sm.getSymReducedParams(network0, include_nonexistent=False)
    outcomes = np.zeros((len(beta_range), 1))
    bounds = [(1e-6, 20) for i in range(2)]

    for i in range(len(lambda_cc_range)):
        print(i)
        for j in range(len(lambda_b_range)):
            A_init = parameterized([1, lambda_cc_range[i]])
            score_ext = gl.KL_score_external(A_init, beta_range[j], network0)
            score_baseline = gl.KL_score(network0, beta_range[j])
            results[i][j] = score_ext/score_baseline
    plt.figure(5)
    plt.rcParams.update({'font.size': 14})
    cax = plt.imshow(results, cmap = 'RdBu',extent=[.01, 1, .01, 1], origin='lower', vmax = 1.1, vmin = .9, aspect = 1, norm = mn.MidpointNormalize(midpoint=1))
    plt.title(r"$\frac{D_{KL}(A || f(A_{in}))}{D_{KL}(A || f(A))}$", size=18)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel(r"$\beta$", size=20)
    plt.ylabel(r"$\lambda _{l}$", size=20)
    cbar = plt.colorbar(cax, ticks = [.9, .95, 1.0, 1.05, 1.1])
    cbar.ax.set_yticklabels(['<.9','.95','1.0','1.05', '> 1.1'])
    plt.tight_layout()

def render_network(input, fignum, graph_pos = None, k = 1, nodecolors = 'lightblue', edgescale = 10, node_size = 50, highlighted = None):
    plt.figure(fignum)
    G_0 = nx.from_numpy_matrix(input)
    d = dict(G_0.degree)
    if not graph_pos:
        graph_pos = nx.spring_layout(G_0, iterations = 1000, k = k)
    edgewidth = [edgescale*d['weight']  for (u, v, d) in G_0.edges(data=True)]
    edgecolor = ['grey' for e in G_0.edges]
    nx.draw_networkx(G_0, graph_pos, width=np.zeros(len(input)), with_labels=False, node_color=nodecolors, node_size=node_size)
    nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, width=edgewidth)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    ax.collections[0].set_linewidth(.5)
    plt.axis('off')
    return graph_pos
