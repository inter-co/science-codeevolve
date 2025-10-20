# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
# Removed direct pdist import as it will be replaced by Numba-optimized version
import numba # Added numba for JIT compilation

# Numba-optimized custom pairwise distance function to replace scipy.spatial.distance.pdist
@numba.jit(nopython=True)
def _numba_pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise Euclidean distances between points,
    returning a condensed distance matrix (like scipy.spatial.distance.pdist).
    Optimized with Numba for performance.
    """
    n_points = points.shape[0]
    if n_points < 2:
        return np.empty(0, dtype=points.dtype)
    
    num_distances = n_points * (n_points - 1) // 2
    dist_array = np.empty(num_distances, dtype=points.dtype)
    
    k = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_array[k] = np.sqrt(dx*dx + dy*dy + dz*dz)
            k += 1
    return dist_array

# Numba-optimized objective function
@numba.jit(nopython=True)
def _objective_func(coordinates_flat: np.ndarray, n_points: int, dim: int) -> float:
    """
    Objective function for the optimizer, optimized with Numba.
    Takes a flattened array of coordinates, reshapes them,
    centers and scales them to fit within a unit sphere,
    and returns the negative of the min/max distance ratio.
    """
    points = coordinates_flat.reshape((n_points, dim))

    # Centering (Numba-compatible)
    centroid = np.empty(dim, dtype=points.dtype)
    for j in range(dim):
        centroid[j] = np.mean(points[:, j])
    centered_points = points - centroid
    
    # Calculate distances from origin for scaling (Numba-compatible)
    max_dist_from_origin = 0.0
    for i in range(n_points):
        dist_sq = 0.0
        for j in range(dim):
            dist_sq += centered_points[i, j]**2
        dist = np.sqrt(dist_sq)
        if dist > max_dist_from_origin:
            max_dist_from_origin = dist

    # Handle edge case where all points are coincident or nearly so.
    if max_dist_from_origin < 1e-9:
        return -1e10 # Penalize heavily

    normalized_points = centered_points / max_dist_from_origin

    # Calculate all pairwise Euclidean distances using Numba-optimized function
    pairwise_distances = _numba_pairwise_distances(normalized_points)

    if len(pairwise_distances) == 0:
        return -1e10 # Penalize heavily if less than 2 points or all collapsed

    dmin = np.min(pairwise_distances)
    dmax = np.max(pairwise_distances)

    if dmax < 1e-9: # Safeguard: if dmax is still very small
        return -1e10 # Penalize heavily

    return -dmin / dmax


def min_max_dist_dim3_14()->np.ndarray:
    """ 
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    This function uses a hybrid global-local optimization strategy with a Numba-accelerated
    objective function to find an optimal arrangement.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points,
                normalized to fit within a unit sphere.
    """

    n = 14 # Number of points
    d = 3  # Dimensions

    np.random.seed(42)

    bounds = [(-1.0, 1.0)] * (n * d)

    best_ratio = -np.inf
    best_points_global = None
    
    # --- Global Search Phase ---
    # Employ multiple independent runs of Dual Annealing. Numba acceleration allows for more iterations.
    num_runs = 5 

    for run_idx in range(num_runs):
        current_seed = 42 + run_idx 
        np.random.seed(current_seed)

        # Generate a fresh, random initial guess on a unit sphere for each run.
        initial_points_on_sphere = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(initial_points_on_sphere, axis=1, keepdims=True)
        # Handle division by zero for points exactly at origin
        initial_points_on_sphere = initial_points_on_sphere / np.where(norms == 0, 1, norms) 
        initial_guess_x0 = initial_points_on_sphere.flatten()

        result = dual_annealing(
            func=_objective_func, # Use Numba-optimized objective
            bounds=bounds,
            args=(n, d),
            x0=initial_guess_x0,
            maxiter=4000, # Increased maxiter due to Numba's speedup, allowing deeper exploration per run.
                          # Total global iterations: 5 runs * 4000 = 20,000.
            initial_temp=52300.0,
            seed=current_seed,
            no_local_search=False,
        )

        current_ratio = -result.fun
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points_global = result.x.reshape((n, d))

    if best_points_global is None:
        return np.zeros((n, d))

    # --- Local Refinement (Polishing) Phase ---
    initial_guess_for_local = best_points_global.flatten()
    
    local_result = minimize(
        fun=_objective_func, # Use Numba-optimized objective for local refinement too
        x0=initial_guess_for_local,
        args=(n, d),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    polished_coords_flat = local_result.x

    # --- Final Normalization ---
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