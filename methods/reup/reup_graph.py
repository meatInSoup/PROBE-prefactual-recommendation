import numpy as np

from methods.reup.chebysev import chebysev_center
from methods.reup.q_determine import exhaustive_search, find_q
from methods.reup.graph import build_graph, shortest_path_graph, eval_cost


def generate_recourse(x0, model, random_state, params=dict()):
    # General parameters
    train_data = params['train_data']
    labels = params['labels']
    data = train_data[labels == 1]

    train_data = np.concatenate([x0.reshape(1, -1), train_data])

    pos_idx = np.where(labels == 1)[0] + 1

    cat_indices = params['cat_indices']

    # Graph parameters
    T = params['reup_params']['T']
    # epsilon = params['reup_params']['eps']
    is_knn = params['reup_params']['knn']
    n = params['reup_params']['n']
    cost_type = params['cost_type']
    w = params.get('w')
    epsilon = np.random.logistic(loc=0, scale=0.05514, size=1)

    # Questions generation
    P, A_opt, mean_rank = find_q(x0, data, T, params['A'], epsilon, cost_correction=True, pair=False, cost_type=cost_type, w=w)

    # Recourse generation
    graph_opt = build_graph(train_data, A_opt, is_knn, n)
    result = shortest_path_graph(graph_opt, pos_idx)
    # Handle cases where no valid path is found
    if result is None or result[2] is None:
        print("No valid path found. Skipping this sample.")
        return None, None, mean_rank, False  # Skip this sample
    path = shortest_path_graph(graph_opt, pos_idx)[2]
    recourse = train_data[path[-1]]
    # graph_iden = build_graph(train_data, np.eye(train_data.shape[1]), is_knn, n)
    cost = eval_cost(params['A'], train_data, path, cost=cost_type, w=w)
    print(cost)
    feasible = True

    return recourse, cost, mean_rank, feasible
