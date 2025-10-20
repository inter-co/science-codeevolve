# EVOLVE-BLOCK-START
import numpy as np
import numba
from scipy.optimize import dual_annealing, minimize
from scipy.stats import qmc

@numba.jit(nopython=True, fastmath=True)
def _pdist_numba(points: np.ndarray) -> np.ndarray:
    """Numba-compatible implementation of pairwise Euclidean distances."""
    n_points = points.shape[0]
    n_distances = n_points * (n_points - 1) // 2
    distances = np.empty(n_distances, dtype=np.float64)
    k = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist_sq = 0.0
            for dim in range(points.shape[1]):
                diff = points[i, dim] - points[j, dim]
                dist_sq += diff * diff
            distances[k] = np.sqrt(dist_sq)
            k += 1
    return distances

@numba.jit(nopython=True, fastmath=True)
def _objective_function_spherical(coordinates_flat: np.ndarray, n_points: int, dim: int) -> float:
    """
    Numba-accelerated objective function that evaluates points on a unit sphere.
    Inspired by Inspiration Program 3.
    """
    points = coordinates_flat.reshape((n_points, dim))

    # Centering (Numba-compatible)
    centroid = np.empty(dim, dtype=points.dtype)
    for j in range(dim):
        centroid[j] = np.mean(points[:, j])
    centered_points = points - centroid
    
    # Scale points to fit within a unit sphere (Numba-compatible)
    max_dist_from_origin = 0.0
    for i in range(n_points):
        dist_sq = 0.0
        for j in range(dim):
            dist_sq += centered_points[i, j]**2
        dist = np.sqrt(dist_sq)
        if dist > max_dist_from_origin:
            max_dist_from_origin = dist

    if max_dist_from_origin < 1e-9:
        return np.inf

    normalized_points = centered_points / max_dist_from_origin
    pairwise_distances = _pdist_numba(normalized_points)

    if len(pairwise_distances) == 0:
        return np.inf

    dmin = np.min(pairwise_distances)
    dmax = np.max(pairwise_distances)

    if dmax < 1e-9:
        return np.inf

    return -dmin / dmax

def min_max_dist_dim3_14()->np.ndarray:
    """ 
    Creates 14 points in 3D by using a hybrid global-local optimization strategy
    to maximize the dmin/dmax ratio on a unit sphere. This approach is adapted
    from the high-performing Inspiration Program 3.
    """
    n = 14
    d = 3
    n_dims = n * d
    optimization_seed = 42

    # Bounds are set to [-1, 1], suitable for optimizing points that will be normalized to a sphere.
    bounds = [(-1.0, 1.0)] * n_dims

    # --- Stage 1: Global Search with Dual Annealing ---
    # Use a Sobol sequence for a high-quality, deterministic initial guess.
    sampler = qmc.Sobol(d=n_dims, seed=optimization_seed)
    sampler.fast_forward(1)
    initial_x0 = (sampler.random(n=1).flatten() * 2) - 1 # Scale from [0,1] to [-1,1]

    global_result = dual_annealing(
        func=_objective_function_spherical,
        bounds=bounds,
        args=(n, d),
        x0=initial_x0,
        maxiter=5000,
        maxfun=15_000_000, # High budget for thorough global exploration
        seed=optimization_seed,
        no_local_search=True # Defer local search to a dedicated polishing stage
    )

    # --- Stage 2: Local Refinement (Polishing) ---
    # Use L-BFGS-B to fine-tune the result from the global search.
    local_result = minimize(
        fun=_objective_function_spherical,
        x0=global_result.x,
        args=(n, d),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # High precision options
    )
    
    polished_coords_flat = local_result.x

    # --- Final Normalization ---
    # Reshape and apply the final normalization to ensure points are on the unit sphere.
    final_points = polished_coords_flat.reshape((n, d))
    centroid = np.mean(final_points, axis=0)
    centered_points = final_points - centroid
    
    distances_from_origin = np.linalg.norm(centered_points, axis=1)
    max_dist_from_origin = np.max(distances_from_origin)

    if max_dist_from_origin < 1e-9:
        return np.zeros((n, d))
        
    normalized_optimized_points = centered_points / max_dist_from_origin

    return normalized_optimized_points
# EVOLVE-BLOCK-END