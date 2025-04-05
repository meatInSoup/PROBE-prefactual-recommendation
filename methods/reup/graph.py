import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
import networkx as nx


def mahalanobis_dist(x, y, A):
    return np.sqrt((x - y).T @ A @ (x - y))


def build_graph(data, A_opt, is_knn, n):
    def dist(x, y):
        return np.sqrt((x - y).T @ A_opt @ (x - y))

    if is_knn:
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric=dist).fit(data)
        graph = nbrs.kneighbors_graph(data, mode="distance").toarray()
    else:
        graph = radius_neighbors_graph(data, radius=n, metric="pyfunc", func=dist, n_jobs=-1)

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
   

if __name__ == '__main__':
    data = np.random.rand(100, 2)
    A = np.random.rand(2, 2)
    A = A @ A.T

    graph = build_graph(data, A, True, 15)
    print(shortest_path_graph(graph))
