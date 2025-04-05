import numpy as np
import scipy
import heapq
from methods.probe import bayesian_utils, bayesian_inference, bayesian_posterior_update
from scipy.special import expit

import torch


def max_entropy_search(A_0, Sigma, m, feasible_set, x_0, history, size=50):
    current_entropy = -np.inf

    for i in range(len(feasible_set) - 1):
        for j in range(i + 1, len(feasible_set)):

            check_history = (i, j) in history

            if not check_history:
                x_i = feasible_set[i].reshape(-1, 1)
                x_j = feasible_set[j].reshape(-1, 1)

                # compute entropy
                entropy_ij = bayesian_utils.entropy_McKay(x_i, x_j, x_0, Sigma, m)

                # compare the entropy
                if current_entropy < entropy_ij:
                    current_entropy = entropy_ij
                    M_ij = bayesian_utils.compute_M(x_i, x_j, x_0)
                    objective = np.trace(A_0 @ M_ij)
                    R_ij = 1 if objective <= 0 else -1
                    result = [
                        {
                            "indices": (i, j),
                            "vector_pair": (x_i - x_0, x_j - x_0),
                            "entropy": entropy_ij,
                            "R_ij": R_ij,
                            "M_ij": M_ij,
                        }
                    ]

    return result


def max_entropy_search_topk(A_0, Sigma, m, feasible_set, x_0, history, k=4):
    """
    Perform a max-entropy search over a feasible set and return the top-k results.

    Parameters:
        A_0: Matrix used in the trace computation.
        Sigma: Covariance matrix.
        m: Parameter for entropy computation.
        feasible_set: List of candidate vectors.
        x_0: Current reference vector.
        history: Set of previously evaluated index pairs (i, j).
        k: Number of top results to return.

    Returns:
        A list of top-k tuples with (index pair, entropy, R_ij, M_ij), sorted by entropy in descending order.
    """
    # Initialize a min-heap to store the top-k results
    top_k_heap = []
    feasible_set = np.array(feasible_set)  # Convert to NumPy array for efficiency

    # Precompute x_0 as a column vector
    x_0 = x_0.reshape(-1, 1)

    # Loop over unique pairs (i, j) in the feasible set
    for i in range(len(feasible_set)):
        x_i = feasible_set[i].reshape(-1, 1)

        for j in range(i + 1, len(feasible_set)):
            if (i, j) in history:  # Skip pairs already in the history
                continue

            x_j = feasible_set[j].reshape(-1, 1)

            # Compute entropy for the pair (x_i, x_j)
            entropy_ij = bayesian_utils.entropy_McKay(x_i, x_j, x_0, Sigma, m)

            # Compute M_ij and related values
            M_ij = bayesian_utils.compute_M(x_i, x_j, x_0)
            objective = np.trace(A_0 @ M_ij)
            R_ij = 1 if objective <= 0 else -1

            # Push the result onto the heap
            heapq.heappush(
                top_k_heap, (entropy_ij, (i, j), (x_i - x_0, x_j - x_0), R_ij, M_ij)
            )

            # Maintain the heap size to be at most k
            if len(top_k_heap) > k:
                heapq.heappop(top_k_heap)

    # Extract and sort the top-k results by entropy in descending order
    top_k_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)

    # Format the output as a list of tuples
    # return [(pair, entropy, R_ij, M_ij) for entropy, pair, R_ij, M_ij in top_k_results]
    return [
        {
            "indices": result[1],
            "vector_pair": result[2],
            "entropy": result[0],
            "R_ij": result[3],
            "M_ij": result[4],
        }
        for result in top_k_results
    ]


def sampling_max_entropy_search(
    A_0,
    Sigma,
    m,
    feasible_set,
    x_0,
    history,
    size=50,
    cost="mahalanobis",
    kappa=10000000,
    w=None, # for weighted-l1
):
    current_entropy = -np.inf

    sampled_index = np.random.choice(
        np.arange(feasible_set.shape[0]),
        size=min(size, feasible_set.shape[0]),
        replace=False,
    )
    
    # Cache sampled points for efficiency
    sampled_points = feasible_set[sampled_index]
    x_0 = x_0.reshape(-1, 1)
    if w is None:
        w = w.reshape(-1, 1)

    for i in range(len(sampled_index) - 1):
        for j in range(i + 1, len(sampled_index)):

            check_history = (sampled_index[i], sampled_index[j]) in history

            if not check_history:
                x_i = feasible_set[sampled_index[i]].reshape(-1, 1)
                x_j = feasible_set[sampled_index[j]].reshape(-1, 1)

                # compute entropy
                entropy_ij = bayesian_utils.entropy_McKay(x_i, x_j, x_0, Sigma, m)

                # compare the entropy
                if current_entropy < entropy_ij:
                    current_entropy = entropy_ij
                    M_ij = bayesian_utils.compute_M(x_i, x_j, x_0)
                    noise = np.random.logistic(loc=0, scale=0.05514, size=1)

                    if cost == "mahalanobis":
                        objective = np.trace(A_0 @ M_ij)
                    elif cost == "l1":
                        l_i = np.linalg.norm((x_i - x_0), ord=1)
                        l_j = np.linalg.norm((x_j - x_0), ord=1)
                        objective = l_i - l_j
                    elif cost == "l2":
                        l_i = np.linalg.norm((x_i - x_0), ord=2)
                        l_j = np.linalg.norm((x_j - x_0), ord=2)
                        objective = l_i - l_j
                    elif cost == "weighted-l1":
                        l_i = np.sum(w * np.abs(x_i -x_0))
                        l_j = np.sum(w * np.abs(x_j -x_0))
                        objective = l_i - l_j
                    objective += noise
                    R_ij = 1 if objective <= 0 else -1

                    result = [
                        {
                            "indices": (i, j),
                            "vector_pair": (x_i - x_0, x_j - x_0),
                            "entropy": entropy_ij,
                            "R_ij": R_ij,
                            "M_ij": M_ij,
                        }
                    ]
                    
                    # result = (
                    #     (sampled_index[i], sampled_index[j]),
                    #     (x_i - x_0, x_j - x_0),
                    #     entropy_ij,
                    #     R_ij,
                    #     M_ij,
                    # )

    return result


def sampling_max_entropy_search_topk(
    A_0,
    Sigma,
    m,
    feasible_set,
    x_0,
    history,
    size=50,
    cost="mahalanobis",
    kappa=1e12,
    top_k=4,
    w=None # for weighted l1
):
    """
    Perform a sampling-based max-entropy search and return the top-k results.

    Parameters:
        A_0: Matrix for trace computation.
        Sigma: Covariance matrix.
        m: Parameter for entropy computation.
        feasible_set: Array of candidate vectors.
        x_0: Current reference vector.
        history: Set of previously evaluated index pairs.
        size: Number of samples to consider.
        cost: Cost metric ('mahalanobis', 'l1-cost', 'l2-cost').
        kappa: Scaling factor for probabilistic decision.
        top_k: Number of top results to return.

    Returns:
        List of top-k tuples, each containing:
        - Index pair (i, j)
        - Entropy value
        - Decision value R_ij
        - Matrix M_ij
        - Epsilon value
    """
    # Sample indices without replacement and ensure unique sampling
    sampled_index = np.random.choice(
        np.arange(feasible_set.shape[0]),
        size=min(size, feasible_set.shape[0]),
        replace=False,
    )

    # Cache sampled points for efficiency
    sampled_points = feasible_set[sampled_index]
    x_0 = x_0.reshape(-1, 1)
    if w is not None:
        w = w.reshape(-1, 1)

    # Initialize a min-heap for the top-k results
    top_k_heap = []

    # Iterate over unique pairs (i, j)
    for idx_i in range(len(sampled_index) - 1):
        x_i = sampled_points[idx_i].reshape(-1, 1)
        for idx_j in range(idx_i + 1, len(sampled_index)):
            i, j = sampled_index[idx_i], sampled_index[idx_j]

            if (i, j) in history:
                continue

            x_j = sampled_points[idx_j].reshape(-1, 1)

            # Compute entropy
            entropy_ij = bayesian_utils.entropy_McKay(x_i, x_j, x_0, Sigma, m)

            # Compute M_ij
            M_ij = bayesian_utils.compute_M(x_i, x_j, x_0)
            noise = np.random.logistic(loc=0, scale=0.05514, size=1)


            # Compute objective based on cost metric
            if cost == "mahalanobis":
                objective = np.trace(A_0 @ M_ij)
            elif cost == "l1":
                objective = np.linalg.norm(x_i - x_0, ord=1) - np.linalg.norm(
                    x_j - x_0, ord=1
                )
            elif cost == "l2":
                objective = np.linalg.norm(x_i - x_0, ord=2) - np.linalg.norm(
                    x_j - x_0, ord=2
                )
            elif cost == "weighted-l1":
                l_i = np.sum(w * np.abs(x_i -x_0))
                l_j = np.sum(w * np.abs(x_j -x_0))
                objective = l_i - l_j
            else:
                raise ValueError(f"Unsupported cost metric: {cost}")
            objective += noise
            
            R_ij = 1 if objective <= 0 else -1

            # Add the result to the heap
            result = (entropy_ij, (i, j), (x_i - x_0, x_j - x_0), R_ij, M_ij)
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, result)
            else:
                heapq.heappushpop(top_k_heap, result)

    # Sort the top-k results by entropy in descending order
    top_k_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)

    # Format the results for clarity
    return [
        {
            "indices": result[1],
            "vector_pair": result[2],
            "entropy": result[0],
            "R_ij": result[3],
            "M_ij": result[4],
        }
        for result in top_k_results
    ]


def random_search(A_0, x_0, feasible_set, history, size):
    full_index = np.arange(feasible_set.shape[0])
    sampled_index = np.random.choice(full_index, size=100)
    sampled_index = np.unique(sampled_index)
    sampled_index = sampled_index[:size]

    check_history = True

    while check_history:
        i, j = np.random.choice(sampled_index, size=2)
        check_history = (i, j) in history or (j, i) in history

    x_i, x_j = feasible_set[i].reshape(-1, 1), feasible_set[j].reshape(-1, 1)
    entropy_ij = None
    M_ij = bayesian_utils.compute_M(x_i, x_j, x_0)
    objective = np.trace(A_0 @ M_ij)
    R_ij = 1 if objective <= 0 else -1
    result = ((i, j), (x_i - x_0, x_j - x_0), entropy_ij, R_ij, M_ij)

    return result
