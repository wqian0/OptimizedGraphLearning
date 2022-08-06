import os
import pickle as pk
from copy import deepcopy
import copy
import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp
from scipy import optimize as op
from scipy import stats
import src.symmetries as sm
import src.GenerateGraph as gg
import src.GraphRender as gr
import matplotlib.pyplot as plt
import sys
head_dir = "C:/Users/billy/PycharmProjects/GraphLearningAgents/"

def process(trial_no, name):
    new_dir = head_dir + name +"/"
    f = open(new_dir + str(trial_no)+ "_" + name + ".txt", "r")
    param_vals = []
    opts = []
    scores_orig = []
    scores = []
    beta = np.linspace(1e-3, 1, 25)[trial_no]

    for x in f.readline().strip().split("\t"):
        param_vals.append(float(x))
    for x in f.readline().strip().split("\t"):
        opts.append(float(x))
    for x in f.readline().strip().split("\t"):
        scores_orig.append(float(x))
    for x in f.readline().strip().split("\t"):
        scores.append(float(x))
    f.close()
    return beta, param_vals, opts, scores_orig, scores

def process_all(name, num_betas, res):
    new_dir = head_dir + name + "/"
    hmap_opts = np.zeros((num_betas, res))
    hmap_scores_orig = np.zeros((num_betas, res))
    hmap_scores = np.zeros((num_betas, res))
    for i in range(num_betas):
        f = open(new_dir + str(i) + "_" + name + ".txt", "r")
        param_vals = []
        opts = []
        scores_orig = []
        scores = []
        for x in f.readline().strip().split("\t"):
            param_vals.append(float(x))
        for x in f.readline().strip().split("\t"):
            opts.append(float(x))
        for x in f.readline().strip().split("\t"):
            scores_orig.append(float(x))
        for x in f.readline().strip().split("\t"):
            scores.append(float(x))
        f.close()
        if len(opts) == len(hmap_opts[i]):
            hmap_opts[i] = opts
            hmap_scores_orig[i] = scores_orig
            hmap_scores[i] = scores
    return param_vals, hmap_opts, hmap_scores_orig, hmap_scores

def process_node_pos(fname, N):
    f = open(fname, "r")
    output = [[] for _ in range(N)]
    for i in range(13):
        f.readline()
    for i in range(N):
        for j in range(7):
            curr = f.readline()
        parts = curr.split(" ")
        if len(parts) == 11:
            return output
        currID = int(parts[11][:-3])
        for j in range(4):
            f.readline()
        x_parts = f.readline().split(" ")
        y_parts = f.readline().split(" ")
        x_coord = float(x_parts[-1][:-2])
        y_coord = float(y_parts[-1][:-1])
        output[currID].append(x_coord)
        output[currID].append(y_coord)
        for j in range(3):
            f.readline()
    return output

if __name__ == '__main__':


    edwards = pk.load(open("all_data.pk", "rb"))
    axler = pk.load(open("all_data_axler.pk","rb"))
    modular = pk.load(open("all_data_modular_10.pk", "rb"))
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
    # betas = np.linspace(1e-3, 1, 25)
    # num_betas = 8
    # name = 'mod'
    # x = np.ones((num_betas, 3))
    # y = np.ones((num_betas, 3))
    #
    # # blue to purple, use 8
    # x[:, 0:3] = (0, .5, .5)
    # y[:, 0:3] = (.5, 0, .5)
    # # green to orange, use 6
    # # x[:, 0:3] = (0, .5, 0)
    # # y[:, 0:3] = (1, .5, 0)
    #
    # #sw colorscheme
    # # x[:, 0:3] = (.75, .25, 0)
    # # y[:, 0:3] = (0, .75, .5)
    # c = np.linspace(0, 1, num_betas)[:, None]
    # gradient = x + (y - x) * c
    #
    # plt.figure(0, figsize= (5.5, 4.5))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel('Fraction of edges within communities')
    # plt.ylabel('Optimal cross-cluster weight')
    # beta, param_vals, opts, scores_orig, scores = process(0, name)
    # plt.scatter(param_vals, opts, label=r"$\beta = 10^{-3}$" , s=30, color=gradient[0], marker = '*')
    # plt.plot(param_vals, opts, color=gradient[0], linewidth = .8)
    # plt.tight_layout()
    # plt.xlim([0.18, .98])
    #
    # for i in range(1, 6):
    #     beta, param_vals, opts, scores_orig, scores = process(i, name)
    #     plt.scatter(param_vals, opts, label=r"$\beta = $" + str(beta)[0:4], s = 30, color = gradient[i], marker = '*')
    #     plt.plot(param_vals, opts, color = gradient[i], linewidth = .8)
    # plt.legend()
    # plt.legend(frameon=False, prop={'size': 12}, labelspacing=.2, handletextpad=0.0, borderpad = 0)
    #
    # plt.figure(1, figsize= (5.5, 4.5))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel('Fraction of edges within communities')
    # plt.ylabel('KL Divergence Ratio')
    # plt.xlim([0.18, .98])
    # plt.ylim([0.4, 1.03])
    #
    # beta, param_vals, opts, scores_orig, scores = process(0, name)
    # plt.scatter(param_vals, np.array(scores) / np.array(scores_orig), label=r"$\beta = 10^{-3}$", s=20,
    #             color=gradient[0])
    # plt.plot(param_vals, np.array(scores) / np.array(scores_orig), color=gradient[0], linewidth = .8)
    #
    # for i in range(1, 6):
    #     beta, param_vals, opts, scores_orig, scores = process(i, name)
    #     plt.scatter(param_vals, np.array(scores)/np.array(scores_orig), label=r"$\beta = $" + str(beta)[0:4], s=20, color = gradient[i])
    #     plt.plot(param_vals, np.array(scores) / np.array(scores_orig), color = gradient[i], linewidth = .8)
    #
    # plt.legend(frameon=False, prop={'size': 12}, labelspacing=.2, handletextpad=0.0, borderpad = 0)
    # plt.tight_layout()
    #
    # plt.figure(2, figsize=(5.5, 4.5))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel('Rewiring probability')
    # plt.ylabel('KL Divergence Ratio')
    # beta, param_vals, opts, scores_orig, scores = process(0, 'SW_updated')
    # plt.scatter(param_vals[7:], (np.array(scores) / np.array(scores_orig))[7:], label=r"$\beta = 10^{-3}$", s=20,
    #             color=gradient[0])
    # plt.plot(param_vals[7:], (np.array(scores) / np.array(scores_orig))[7:], color = gradient[0], linewidth = .7)
    # for i in range(1,6):
    #     beta, param_vals, opts, scores_orig, scores = process(i, "SW_updated")
    #     plt.scatter(param_vals[7:], np.array(scores[7:])/ np.array(scores_orig[7:]), label=r"$\beta = $" + str(beta)[0:4], s=20,
    #                 color=gradient[i])
    #     plt.plot(param_vals[7:], np.array(scores[7:]) / np.array(scores_orig[7:]),
    #                 color=gradient[i], linewidth = .8)
    #     plt.xscale('log')
    # plt.legend(frameon=False, prop={'size': 12}, labelspacing=.2, handletextpad=0, borderpad = 0,  loc = 3)
    # plt.tight_layout()
    #
    # plt.figure(3, figsize=(5.5, 4.5))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel('Rewiring probability')
    # plt.ylabel('Optimal non-ring weight')
    # beta, param_vals, opts, scores_orig, scores = process(0, 'SW_updated')
    # plt.scatter(param_vals[7:], opts[7:], label=r"$\beta = 10^{-3}$", s=30, color=gradient[0], marker='*', linewidth = .7)
    # plt.plot(param_vals[7:], opts[7:], linewidth=.7, color=gradient[0])
    # plt.xscale('log')
    # for i in range(1, 6):
    #     beta, param_vals, opts, scores_orig, scores = process(i, "SW_updated")
    #     plt.scatter(param_vals[7:], opts[7:], label=r"$\beta = $" + str(beta)[0:4], s=30, color = gradient[i], marker='*')
    #     plt.plot(param_vals[7:], opts[7:], linewidth =.7, color = gradient[i])
    # plt.legend(frameon=False, prop={'size': 12}, labelspacing=.2, handletextpad=0, borderpad = 0)
    # plt.tight_layout()
    # plt.show()
