import numpy as np
import src.GenerateGraph as gg
from src.GraphLearning import optimize_one_param_learnability

'''
Functions for running Watts-Strogatz and Stochastic Block Model one-parameter optimization trials
'''
def WS_trials(N_tot, k, p_res, trials, beta):
    with np.errstate(invalid='ignore'):
        p_vals = np.logspace(0, 4, p_res)
        p_vals /= p_vals[-1]
        print(p_vals)
        opts = np.zeros(p_res)
        scores_orig =  np.zeros(p_res)
        scores_s = np.zeros(p_res)
        for i in range(p_res):
            for j in range(trials):
                network, parameterized = gg.small_world_parameterized(N_tot, k, p_vals[i])
                network_s, opt, score_orig, score_s = optimize_one_param_learnability(network, parameterized, beta)
                opts[i] += opt
                scores_orig[i] += score_orig
                scores_s[i] += score_s
        opts /= trials
        scores_orig /= trials
        scores_s /= trials
        return p_vals, opts, scores_orig, scores_s

def SBM_trials(N_tot, N_comm, edges, alpha, frac_res, trials, beta):
    with np.errstate(invalid='ignore'):
        frac_modules = np.linspace(.2, 1, frac_res, endpoint = False)
        mod_opts, hMod_opts = np.zeros(frac_res), np.zeros(frac_res)
        scores_mod_orig, scores_hMod_orig = np.zeros(frac_res), np.zeros(frac_res)
        scores_mod_s, scores_hMod_s = np.zeros(frac_res), np.zeros(frac_res)
        for i in range(frac_res):
            for j in range(trials):
                mod, parMod = gg.get_random_modular(N_tot, N_comm, edges, frac_modules[i])
                hMod, parhMod = gg.get_hierarchical_modular(N_tot, N_comm, edges, frac_modules[i], alpha)
                mod_s, mod_opt, score_mod_orig, score_mod_s = optimize_one_param_learnability(mod, parMod, beta)
                hMod_s, hMod_opt, score_hMod_orig, score_hMod_s = optimize_one_param_learnability(hMod, parhMod, beta)
                mod_opts[i] += mod_opt
                hMod_opts[i] += hMod_opt
                scores_mod_orig[i] += score_mod_orig
                scores_hMod_orig[i] += score_hMod_orig
                scores_mod_s[i] += score_mod_s
                scores_hMod_s[i] += score_hMod_s
        mod_opts /= trials
        hMod_opts /= trials
        scores_mod_orig /= trials
        scores_hMod_orig /= trials
        scores_mod_s /= trials
        scores_hMod_s /= trials
        return frac_modules, mod_opts, hMod_opts, scores_mod_orig, scores_hMod_orig, scores_mod_s, scores_hMod_s