import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')


import numpy as np
import scipy
import heapq
from methods.probe import bayesian_utils, bayesian_posterior_update, question_selection

import torch

from methods.reup.chebysev import chebysev_center, sdp_cost


# For Bayesian

def bayesian_PE(
    A_0,
    Sigma,
    m,
    x_0,
    feasible_set,
    sessions,
    iterations,
    lr,
    tau,
    size=50,
    pair=True,
    cost_type="mahalanobis",
    w=None
):
    # pair=True then we use top1 otherwise use topk
    log_dict = {}
    lst_ind = []
    lst_responses = []
    lst_pos_Sigma = []
    lst_pos_m = []
    lst_mean_rank = []

    # data-dim
    d = x_0.shape[0]

    # initialize the parameter of prior and posterior
    prior_Sigma = Sigma
    prior_m = m
    post_Sigma, post_m = None, None

    for s in range(sessions):
        # create set_m
        set_m = np.arange(d, prior_m + 1)

        # result = random_search(A_0, x_0, feasible_set, lst_ind, size)
        if pair == True:
            kappa = 100
            result = question_selection.sampling_max_entropy_search(
                A_0, prior_Sigma, prior_m, feasible_set, x_0, lst_ind, size, cost=cost_type, kappa=kappa, w=w
            )
            lst_ind.append(result[0]["indices"])
        else:
            kappa = 100
            result = question_selection.sampling_max_entropy_search_topk(
                A_0, prior_Sigma, prior_m, feasible_set, x_0, lst_ind, size, cost=cost_type, kappa=kappa, w=w
            )
            lst_ind.extend([item["indices"] for item in result])
            


        # posterior inference for Sigma and m
        post_Sigma, post_m, losses = bayesian_posterior_update.posterior_inference(
            set_m, result, prior_Sigma, prior_m, tau, d, iterations, lr
        )

        # update the prior from the obtained posterior
        prior_Sigma = post_Sigma
        prior_m = post_m

        # logging
        lst_responses.extend([item["R_ij"] for item in result])
        lst_pos_Sigma.append(post_Sigma)
        lst_pos_m.append(post_m)

        log_s = {
            "iterations": iterations,
            "losses": losses,
        }
        log_dict[s] = log_s
        A_opt = post_m * post_Sigma
        # rank = mean_rank(feasible_set, x_0, A_0, A_opt, 5)
        # lst_mean_rank.append(rank)
        
        # print(A_opt)

    log_dict["lst_ind"] = lst_ind
    log_dict["lst_responses"] = lst_responses
    log_dict["lst_Sigma"] = lst_pos_Sigma
    log_dict["lst_m"] = lst_pos_m
    log_dict["lst_mean_rank"] = None

    return post_Sigma, post_m, log_dict


def bayesian_PE_diag(
    A_0,
    Sigma,
    m,
    x_0,
    feasible_set,
    sessions,
    iterations,
    lr,
    tau,
    size=50,
    pair=True,
    cost_type="mahalanobis",
    w=None
):
    # pair=True then we use top1 otherwise use topk
    log_dict = {}
    lst_ind = []
    lst_responses = []
    lst_pos_Sigma = []
    lst_pos_m = []
    lst_mean_rank = []

    # data-dim
    d = x_0.shape[0]

    # initialize the parameter of prior and posterior
    prior_Sigma = Sigma
    prior_m = m
    post_Sigma, post_m = None, None

    for s in range(sessions):
        # create set_m
        set_m = np.arange(d, prior_m + 1)

        # result = random_search(A_0, x_0, feasible_set, lst_ind, size)
        if pair == True:
            kappa = 50
            result = question_selection.sampling_max_entropy_search(
                A_0, prior_Sigma, prior_m, feasible_set, x_0, lst_ind, size, cost=cost_type, kappa=kappa, w=w
            )
            lst_ind.append(result[0]["indices"])
        else:
            kappa = 50
            result = question_selection.sampling_max_entropy_search_topk(
                A_0, prior_Sigma, prior_m, feasible_set, x_0, lst_ind, size, cost=cost_type, kappa=kappa, w=w
            )
            lst_ind.extend([item["indices"] for item in result])
            


        # posterior inference for Sigma and m
        post_Sigma, post_m, losses = bayesian_posterior_update.posterior_inference_diag(
            set_m, result, prior_Sigma, prior_m, tau, d, iterations, lr
        )

        # update the prior from the obtained posterior
        prior_Sigma = post_Sigma
        prior_m = post_m

        # logging
        lst_responses.extend([item["R_ij"] for item in result])
        lst_pos_Sigma.append(post_Sigma)
        lst_pos_m.append(post_m)

        log_s = {
            "iterations": iterations,
            "losses": losses,
        }
        log_dict[s] = log_s
        A_opt = post_m * post_Sigma
        # rank = mean_rank(feasible_set, x_0, A_0, A_opt, 5)
        # lst_mean_rank.append(rank)
        
        # print(A_opt)

    log_dict["lst_ind"] = lst_ind
    log_dict["lst_responses"] = lst_responses
    log_dict["lst_Sigma"] = lst_pos_Sigma
    log_dict["lst_m"] = lst_pos_m
    log_dict["lst_mean_rank"] = None

    return post_Sigma, post_m, log_dict


def mahalanobis_distance(x, x_0, A):
    delta = x - x_0
    return np.dot(np.dot(delta.T, A), delta)


def compute_ranks(dataset, x_0, A):
    distances = [mahalanobis_distance(x, x_0, A) for x in dataset]
    return np.argsort(distances) + 1


def mean_rank(dataset, x_0, A_0, A_opt, K):
    N = len(dataset)
    ranks_A0 = compute_ranks(dataset, x_0, A_0)
    ranks_A_opt = compute_ranks(dataset, x_0, A_opt)

    top_k_indices = np.argsort(ranks_A_opt)[:K]
    r_i = np.array([ranks_A0[idx] for idx in top_k_indices])

    r_min = (K + 1) * K // 2
    r_max = (2 * N - K + 1) * K // 2
    mean_rank_value = (r_i.sum() - r_min) / r_max
    return mean_rank_value



if __name__ == "__main__":
    A = np.random.rand(2, 2)
    A = A @ A.T
    A_init = np.random.rand(2, 2)
    # Sigma_init = A_init @ A_init.T
    x_0 = np.random.rand(2)
    x_init = np.array([0, 0])
    Sigma_init = np.eye(2)
    data = np.random.rand(100, 2)
    m = 4
    Sigma, post_m, log_dict = bayesian_PE(A, Sigma_init, m, x_0, data, 10, iterations=500, lr=0.5, tau=0.5, pair=False)