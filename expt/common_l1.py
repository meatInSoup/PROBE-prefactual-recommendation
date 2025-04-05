import numpy as np
import scipy
import copy
import os
import torch
import joblib
import sklearn
from functools import partialmethod
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from collections import defaultdict, namedtuple
from time import time
from joblib import Parallel, delayed

import dice_ml

from utils import helpers
from utils.data_transformer import DataTransformer
from utils.funcs import compute_max_distance, lp_dist, compute_validity, compute_proximity, compute_diversity, compute_distance_manifold, compute_dpp, compute_likelihood, compute_pairwise_cosine, compute_kde, compute_proximity_graph, compute_proximity_graph_, compute_diversity_path, hamming_distance, levenshtein_distance, jaccard, mahalanobis_dist, weighted_l1_dist

from classifiers import mlp, random_forest

from methods.face import face
from methods.dice import dice
from methods.reup import reup, reup_graph, reup_graph_iden, reup_graph_gt
from methods.probe import probe, probe_diag, probe_robust
from methods.wachter import wachter, wachter_gt


Results = namedtuple("Results", ["l1_cost", "valid", "rank", "feasible", "log_dict"])
Results_graph = namedtuple("Results_graph", ["valid", "l1_cost", "diversity", "dpp", "manifold_dist", "hamming",  "lev", "jac", "feasible"])


def to_numpy_array(lst):
    pad = len(max(lst, key=len))
    return np.array([i + [0]*(pad-len(i)) for i in lst])


def load_models(dname, cname, wdir):
    pdir = os.path.dirname(wdir)
    pdir = os.path.join(pdir, 'checkpoints')
    models = helpers.pload(f"{cname}_{dname}.pickle", pdir)
    return models


def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.predict(x)
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds)


def enrich_training_data(num_samples, train_data, cat_indices, rng):
    rng = check_random_state(rng)
    cur_n, d = train_data.shape
    min_f_val = np.min(train_data, axis=0)
    max_f_val = np.max(train_data, axis=0)
    new_data = rng.uniform(min_f_val, max_f_val, (num_samples - cur_n, d))

    new_data[:, cat_indices] = new_data[:, cat_indices] >= 0.5

    new_data = np.vstack([train_data, new_data])
    return new_data


def to_mean_std(m, s, is_best):
    if is_best:
        return "\\textbf{" + "{:.2f}".format(m) + "}" + " $\pm$ {:.2f}".format(s)
    else:
        return "{:.2f} $\pm$ {:.2f}".format(m, s)


def _run_single_instance(idx, method, x0, model, seed, logger, params=dict()):
    torch.manual_seed(seed+2)
    np.random.seed(seed+1)
    random_state = check_random_state(seed)

    rank_l = []
    # l1_cost = np.zeros(params['num_w'])
    l1_cost = []
    log_dict_l = []
    feasible = None
    t0 = time()

    if method == dice or method == wachter:
        x_ar, feasible = method.generate_recourse(x0, model, random_state, params)

    # Define a function that wraps the loop logic for each iteration i
    def process_iteration(i):
        params['w'] = params['all_w'][i]
        local_rank_l = []
        local_log_dict_l = []
        local_l1_cost = 0

        if method == reup:
            x_ar, rank, feasible = method.generate_recourse(x0, model, random_state, params)
            local_rank_l.append(rank)
        elif method == reup_graph:
            x_ar, cost, rank, feasible = method.generate_recourse(x0, model, random_state, params)
            local_l1_cost = cost
        elif method == reup_graph_iden or method == reup_graph_gt:
            x_ar, cost, feasible = method.generate_recourse(x0, model, random_state, params)
            local_l1_cost = cost
        elif method == probe:
            x_ar, cost, feasible, log_dict = method.generate_recourse(x0, model, params)
            rank = log_dict["lst_mean_rank"]
            local_rank_l.append(rank)
            local_l1_cost = cost
            local_log_dict_l.append(log_dict)
        elif method == probe_robust:
            x_ar, cost, feasible, log_dict = method.generate_recourse(x0, model, params)
            rank = log_dict["lst_mean_rank"]
            local_rank_l.append(rank)
            local_l1_cost = cost
            local_log_dict_l.append(log_dict)
        elif method == probe_diag:
            x_ar, cost, feasible, log_dict = method.generate_recourse(x0, model, params)
            rank = log_dict["lst_mean_rank"]
            local_rank_l.append(rank)
            local_l1_cost = cost
            local_log_dict_l.append(log_dict)
        elif method == face:
            x_ar, cost, feasible, log_dict = method.generate_recourse(x0, model, random_state, params)
            local_l1_cost = cost
            local_log_dict_l.append(log_dict)
        elif method != dice and method != wachter:
            x_ar, feasible = method.generate_recourse(x0, model, random_state, params)
        if method not in [face, reup_graph, reup_graph_iden, probe, probe_robust, probe_diag]:
            local_l1_cost = weighted_l1_dist(x_ar, x0, params['w'])

        return local_l1_cost, local_rank_l, local_log_dict_l

    # Parallelize the loop with joblib.Parallel and collect the results
    results = Parallel(n_jobs=len(params['all_w']), backend='loky')(
        delayed(process_iteration)(i) for i in range(params['num_w'])
    )

    # Collect all the parallel results
    for res in results:
        local_l1_cost, local_rank_l, local_log_dict_l = res
        l1_cost.append(local_l1_cost)
        rank_l.extend(local_rank_l)
        log_dict_l.extend(local_log_dict_l)

    if params['reup_params']['rank'] or params['probe_params']['rank']:
        rank_l = np.array(rank_l)

    rank = None

    valid = 1.0  # Assuming prediction logic is handled elsewhere

    return Results(l1_cost, valid, rank, feasible, log_dict_l)


method_name_map = {
    'face': "FACE",
    'dice': 'DiCE',
    'dice_ga': 'DICE_GA',
    'gs': "GS",
    'reup': "ReAP-K",
    'pair': "ReAP-2",
    'reup_graph': "ReUP",
    'reup_graph_iden': "ReUP($T=0$)",
    'reup_graph_gt': "ReUP($T$)",
    "probe": "PROBE",
    "probe_robust": "PROBE-Robust",
    "probe_diag": "PROBE-Diag",
    'wachter': "Wachter",
    'gt': "GT",
}


dataset_name_map = {
    "synthesis": "Synthetic data",
    "german": "German",
    "sba": "SBA",
    "bank": "Bank",
    "student": "Student",
    "adult": "Adult",
    "compas": "Compas",
    "gmc": "GiveMeCredit"
}

metric_order = {'cost': -1, 'valid': 1}

metric_order_graph = {'cost': -1, 'valid': 1, 'diversity': -1, 'dpp': 1, 'hamming': 1, 'lev': 1, 'jac': -1}

method_map = {
    "face": face,
    "dice": dice,
    "reup": reup,
    "reup_graph": reup_graph,
    "reup_graph_iden": reup_graph_iden,
    "reup_graph_gt": reup_graph_gt,
    "probe": probe,
    "probe_robust": probe_robust,
    "probe_diag": probe_diag,
    "wachter": wachter,
    "gt": wachter_gt,
}


clf_map = {
    "net0": mlp.Net0,
    "mlp": mlp.Net0,
    "rf": random_forest.RandomForest,
}


train_func_map = {
    'net0': mlp.train,
    'mlp': mlp.train,
    'rf': random_forest.train,
}


synthetic_params = dict(num_samples=1000,
                        x_lim=(-2, 4), y_lim=(-2, 7),
                        f=lambda x, y: y >= 1 + x + 2*x**2 + x**3 - x**4,
                        random_state=42)


synthetic_params_mean_cov = dict(num_samples=1000, mean_0=None, cov_0=None, mean_1=None, cov_1=None, random_state=42)
