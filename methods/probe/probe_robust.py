import numpy as np
import pickle

from methods.probe import q_determine, question_scheme
from methods.probe import graph, bayesian_utils


def generate_recourse(x_0, model, params=dict()):
    # adjust the dimensionality
    x_0 = x_0.reshape(-1, 1)
    dim = x_0.shape[0]

    dname = params["dataset_name"]
    if dname == "synthesis" or dname == "german":
        size = 50
    else:
        size = 25

    # General parameters
    train_data = params["train_data"]
    labels = params["labels"]
    feasible_set = train_data[labels == 1]

    train_data = np.concatenate([x_0.reshape(1, -1), train_data])
    pos_idx = np.where(labels == 1)[0] + 1

    CONST = 4
    prior_m = dim + CONST

    sessions = params["probe_params"]["sessions"]
    iterations = params["probe_params"]["iterations"]
    lr = params["probe_params"]["lr"]
    n_neighbors = params["probe_params"]["n_neighbors"]
    cost_type = params["cost_type"]
    w = params.get('w')
    prior_Sigma = np.eye(dim)

    TAU = 0.1
    alpha = 0.1

    # Preference elicitation + Bayesian inference

    post_Sigma, post_m, log_dict = question_scheme.bayesian_PE(params["A"], prior_Sigma, prior_m, x_0, feasible_set, sessions, iterations, lr, tau=TAU, size=size, pair=False, cost_type=cost_type, w=w)

    # Graph-based recourse generation
    graph_opt = graph.bayesian_build_graph(train_data, post_Sigma, post_m, n_neighbors, diag=False)
    result = graph.shortest_path_graph_robust(train_data, graph_opt, post_Sigma, x_0, pos_idx, alpha)
    # Handle cases where no valid path is found
    if result is None or result[2] is None:
        print("No valid path found. Skipping this sample.")
        return None, None, False, log_dict  # Skip this sample
    path = result[2]
    recourse = train_data[path[-1]]
    cost = graph.eval_cost(params["A"], train_data, path, cost=cost_type, w=w)
    print(cost)
    feasible = True

    # logging
    log_dict["recourse"] = recourse
    log_dict["path"] = path
    log_dict["cost"] = cost
    log_dict["x_0"] = x_0
    log_dict["A_0"] = params["A"]
    log_dict["w"] = w

    return recourse, cost, feasible, log_dict
