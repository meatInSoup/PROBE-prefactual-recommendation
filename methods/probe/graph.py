import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
import networkx as nx
import cvxpy as cp


def bayesian_build_graph(data, pos_Sigma, pos_m, n_neighbors, diag=False):

    if diag:
        eval_Sigma = np.eye(pos_Sigma.shape[0])
        np.fill_diagonal(eval_Sigma, pos_Sigma.diagonal())
    else:
        eval_Sigma = pos_Sigma

    def expected_mahalanobis(x_i, x_j):
        x_i = x_i.reshape(-1, 1)
        x_j = x_j.reshape(-1, 1)
        expected_A = eval_Sigma
        return np.sqrt((x_i - x_j).T @ expected_A @ (x_i - x_j))

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric=expected_mahalanobis
    ).fit(data)
    graph = nbrs.kneighbors_graph(data, mode="distance").toarray()
    # graph = graph + graph.T

    # inds = nbrs.kneighbors_graph(data, mode='connectivity').toarray()
    # inds = inds + inds.T

    return graph


def mahalanobis_dist(x, y, A):
    return np.sqrt((x - y).T @ A @ (x - y))


def build_graph(data, A_opt, is_knn, n):
    def dist(x, y):
        return np.sqrt((x - y).T @ A_opt @ (x - y))

    if is_knn:
        nbrs = NearestNeighbors(n_neighbors=n, algorithm="ball_tree", metric=dist).fit(
            data
        )
        graph = nbrs.kneighbors_graph(data, mode="distance").toarray()
    else:
        graph = radius_neighbors_graph(
            data, radius=n, metric="pyfunc", func=dist, n_jobs=-1
        )
    return graph

def build_graph_binary(data, A_opt, is_knn, n):
    def dist(x, y):
        return np.sqrt((x - y).T @ A_opt @ (x - y))

    if is_knn:
        nbrs = NearestNeighbors(n_neighbors=n, algorithm="ball_tree", metric=dist).fit(
            data
        )
        graph = nbrs.kneighbors_graph(data, mode="connectivity").toarray()
    else:
        graph = radius_neighbors_graph(
            data, radius=n, metric="pyfunc", func=dist, n_jobs=-1
        )
    return graph


def shortest_path_graph(adj, idx):
    """
    Find the shortest path from node 0 to any node in idx after pruning.
    Pruning removes nodes in idx that don't have edges to nodes outside of idx.
    """
    # Convert adjacency matrix to a graph
    G = nx.from_numpy_array(adj)
    
    # Create a list of all nodes in the graph
    all_nodes = list(range(len(adj)))
    
    # Identify nodes not in idx
    non_idx_nodes = [node for node in all_nodes if node not in idx]
    
    # Find nodes in idx that don't have connections to non-idx nodes
    nodes_to_remove = []
    for node in idx:
        # Get all neighbors of this node
        neighbors = list(G.neighbors(node))
        # Check if there are any non-idx nodes in the neighbors
        if not any(neighbor in non_idx_nodes for neighbor in neighbors):
            nodes_to_remove.append(node)
    
    # Remove the identified nodes
    for node in nodes_to_remove:
        G.remove_node(node)
    
    # Create a new idx list without the removed nodes
    pruned_idx = [node for node in idx if node not in nodes_to_remove]
    
    # If node 0 was removed or no valid idx nodes remain, return None
    if 0 not in G.nodes() or not pruned_idx:
        print("No valid paths found after pruning. Skipping this sample.")
        return None
    
    # Compute shortest path from node 0 to all reachable nodes
    try:
        dist_map, path_map = nx.single_source_dijkstra(G, source=0, weight="weight")
    except nx.NetworkXNoPath:
        print("No paths from node 0 to any target node. Skipping this sample.")
        return None
    
    # Filter distances for valid pruned_idx nodes
    valid_nodes = [i for i in pruned_idx if i in dist_map]
    
    if not valid_nodes:  # If no valid paths exist
        print("No valid paths found. Skipping this sample.")
        return None
    
    # Get distances and find the minimum
    dist_l = np.array([dist_map[i] for i in valid_nodes])
    min_idx = valid_nodes[np.argmin(dist_l)]
    dist = dist_map[min_idx]
    path = path_map[min_idx]
    
    return dist, min_idx, path

def compute_M_matrices(data, x0, graph):
    num_nodes = data.shape[0]
    M_matrices = {}

    for i in range(num_nodes):
        for j in range(num_nodes):
            if graph[i, j] > 0:  # Edge exists between node i and node j
                x_i = data[i].reshape(-1, 1)  # Node i's feature vector as a column
                x_j = data[j].reshape(-1, 1)  # Node j's feature vector as a column
                x0_vec = x0.reshape(-1, 1)    # Reference node feature vector as a column
                
                # Compute M_ij
                M_ij = (x_i - x0_vec) @ (x_i - x0_vec).T - (x_j - x0_vec) @ (x_j - x0_vec).T
                M_matrices[(i, j)] = M_ij

    return M_matrices

def shortest_path_graph_robust(data, graph, sigma, x0, index_list, alpha):
    """
    Solves the given graph optimization problem with constraints using CVXPY.
    
    Parameters:
    - graph: np.ndarray, adjacency matrix of the graph.
    - M: list of np.ndarray, edge-specific matrices.
    - Sigma_T: np.ndarray, positive semi-definite matrix for the problem.
    - alpha: float, regularization parameter.
    - x0: int, index of the starting node.
    - C: list or np.ndarray, indicator vector where C[i] = 1 if node i is a target node, 0 otherwise.
    
    Returns:
    - z_opt: np.ndarray, optimal values of z on the edges.
    """
    M_matrices = compute_M_matrices(data, x0, graph)
    num_edges = len(M_matrices)
    num_nodes = graph.shape[0]
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if graph[i, j] > 0]  # All edges
    
    # Binary variables for each edge
    z = cp.Variable(num_edges, boolean=True)
    
    # Define the linear term in the objective
    Sigma_T = sigma.T
    linear_term = cp.sum([
        cp.trace(M_matrices[edge] @ Sigma_T) * z[idx] 
        for idx, edge in enumerate(M_matrices)
    ])
    
    # Define the quadratic term in the objective
    summed_term = cp.sum([
        M_matrices[edge] * z[idx]
        for idx, edge in enumerate(M_matrices)
    ], axis=0)
    quadratic_term = 2 * alpha * cp.trace(summed_term @ Sigma_T @ summed_term @ Sigma_T)
    
     # Objective function
    objective = cp.Minimize(linear_term + quadratic_term)
    
    # Constraints
    constraints = []

    # Constraint 1: Exactly one edge originates from x0
    constraints.append(cp.sum([
        z[idx] for idx, edge in enumerate(M_matrices) if edge[0] == x0
    ]) == 1)

    # Constraint 2: Flow conservation for nodes NOT in index_list
    for xi in range(num_nodes):
        if xi not in index_list and xi != 0:  # Exclude x0
            in_edges = [k for k, (i, j) in enumerate(edges) if j == xi]
            out_edges = [k for k, (i, j) in enumerate(edges) if i == xi]
            constraints.append(cp.sum([z[k] for k in out_edges]) - cp.sum([z[k] for k in in_edges]) == 0)

    # Constraint 3: For every node in index_list, exactly one edge is incident
    for xi in index_list:
        node_edges = [k for k, (i, j) in enumerate(edges) if i == xi or j == xi]
        constraints.append(cp.sum([z[k] for k in node_edges]) == 1)
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)  # You can use MOSEK or another solver
    
    # Get the optimal z values
    z_opt = z.value
    return z_opt

def bayesian_build_graph_linearized(data, pos_Sigma, pos_m, n_neighbors, alpha, current_path, diag=False):
    if diag:
        eval_Sigma = np.eye(pos_Sigma.shape[0])
        np.fill_diagonal(eval_Sigma, pos_Sigma.diagonal())
    else:
        eval_Sigma = pos_Sigma

    def expected_mahalanobis(x_i, x_j):
        x_i = x_i.reshape(-1, 1)
        x_j = x_j.reshape(-1, 1)
        sum_M = np.zeros(eval_Sigma.shape)
        l = len(current_path)
        for i in range(l-1):
            vec1 = data[current_path[i + 1]].reshape(-1, 1)
            vec2 = data[current_path[i]].reshape(-1, 1)
            sum_M += (vec1 - vec2) @ (vec1 - vec2).T
        expected_A = eval_Sigma + 2 * alpha * eval_Sigma @ sum_M @ eval_Sigma
        return np.sqrt((x_i - x_j).T @ expected_A @ (x_i - x_j))

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric=expected_mahalanobis
    ).fit(data)
    graph = nbrs.kneighbors_graph(data, mode="distance").toarray()

    return graph
    


def eval_cost(A, data, path, cost="mahalanobis", w=None):
    l = len(path)
    if w is not None:
        w = w.reshape(-1, 1)
    res = 0
    if cost == "mahalanobis":
        for i in range(l - 1):
            cost = np.sqrt(
                (data[path[i + 1]] - data[path[i]]).T
                @ A
                @ (data[path[i + 1]] - data[path[i]])
            )
            res += cost
    elif cost == "l1":
        for i in range(l - 1):
            cost = np.linalg.norm(data[path[i + 1]] - data[path[i]], ord=1)
            res += cost
    elif cost == "weighted-l1":
        for i in range(l - 1):
            node1 = data[path[i + 1]].reshape(-1, 1)
            node2 = data[path[i]].reshape(-1, 1)
            cost = np.sum(w * np.abs(node1 - node2))
            res += cost

    return res


if __name__ == "__main__":
    data = np.random.rand(100, 2)
    A = np.random.rand(2, 2)
    A = A @ A.T

    graph = build_graph(data, A, True, 15)
    print(shortest_path_graph(graph))
