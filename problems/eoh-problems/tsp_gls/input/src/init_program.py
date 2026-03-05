# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        updated_edge_distance[a, b] *= (1 + edge_n_used[a, b])
        updated_edge_distance[b, a] = updated_edge_distance[a, b]
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    updated_edge_distance[a, b] *= (1 + edge_n_used[a, b])
    updated_edge_distance[b, a] = updated_edge_distance[a, b]
    return updated_edge_distance
# EVOLVE-BLOCK-END
