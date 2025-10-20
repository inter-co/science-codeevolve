# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import numba
from scipy.stats import qmc

# Global constants for the problem, aligned with Inspiration 2
N_POINTS = 16
N_DIMS = 2
N_PARAMS = N_POINTS * N_DIMS # Total number of parameters (16 points * 2 dimensions = 32)

# Numba-optimized pairwise distance calculation
@numba.jit(nopython=True, fastmath=True)
def _pairwise_distances_numba(points: np.ndarray, n: int, d: int) -> np.ndarray:
    """
    Calculates all pairwise Euclidean distances between n points in d dimensions.
    Optimized with Numba for performance.
    """
    num_pairs = n * (n - 1) // 2
    distances = np.empty(num_pairs, dtype=points.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for dim in range(d):
                dist_sq += (points[i, dim] - points[j, dim])**2
            distances[k] = np.sqrt(dist_sq)
            k += 1
    return distances

# Cache for pdist indices to avoid re-computation
_pdist_indices_cache = None
def _get_pdist_indices_cached(n):
    global _pdist_indices_cache
    if _pdist_indices_cache is None or len(_pdist_indices_cache) != n * (n - 1) // 2:
        # Changed dtype to int64 for robustness, as seen in Inspiration 1
        indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
        _pdist_indices_cache = indices
    return _pdist_indices_cache

# Numba-optimized gradient of distances w.r.t. coordinates
@numba.jit(nopython=True, fastmath=True)
def _distances_gradients_numba(points: np.ndarray, n: int, d: int, distances: np.ndarray, pdist_indices: np.ndarray) -> np.ndarray:
    num_distances = len(distances)
    grad_d_matrix = np.zeros((num_distances, n * d), dtype=np.float64)
    diffs = points[pdist_indices[:, 0]] - points[pdist_indices[:, 1]]
    
    for i in range(num_distances):
        # Add a small epsilon to distances for stability in case of division by zero
        dist_inv = 1.0 / (distances[i] + 1e-12) 
        for dim in range(d):
            grad_val = diffs[i, dim] * dist_inv
            grad_d_matrix[i, pdist_indices[i, 0] * d + dim] = grad_val
            grad_d_matrix[i, pdist_indices[i, 1] * d + dim] = -grad_val
            
    return grad_d_matrix

# Numba-optimized exact objective function for global search
@numba.jit(nopython=True, fastmath=True)
def _exact_objective_function_numba(x: np.ndarray, n: int, d: int) -> float:
    """
    Calculates the negative of the exact min_max_ratio.
    This Numba-optimized function is non-smooth and used for global optimization algorithms
    like Differential Evolution. The goal is to minimize this value.
    """
    points = x.reshape(n, d)
    distances = _pairwise_distances_numba(points, n, d)
    
    if len(distances) == 0:
        return np.inf # Penalize heavily
    
    dmin = np.min(distances)
    dmax = np.max(distances)
    
    # Penalize if points collapse or are too close
    if dmin < 1e-9: # Duplicate or near-duplicate points
        return np.inf # High penalty
    if dmax < 1e-9: # All points are essentially at the same location
        return np.inf # High penalty

    return -dmin / dmax

# Numba-optimized smooth objective function with analytical gradient for local refinement
@numba.jit(nopython=True, fastmath=True)
def _smooth_objective_and_grad_numba(x: np.ndarray, n: int, d: int, k: float, pdist_indices: np.ndarray):
    """
    Calculates the negative of the ratio of a smooth minimum distance to a
    smooth maximum distance using LogSumExp approximation, and its analytical gradient.
    Optimized with Numba for performance.
    """
    points = x.reshape(n, d)
    distances = _pairwise_distances_numba(points, n, d)

    if np.min(distances) < 1e-10:
        return np.inf, np.zeros_like(x) # Return penalty with zero gradient

    # Manual Numba-compatible LogSumExp implementation for smooth_max
    scaled_distances_max = k * distances
    c_max = np.max(scaled_distances_max)
    if not np.isfinite(c_max): return np.inf, np.zeros_like(x)
    exp_k_distances_centered = np.exp(scaled_distances_max - c_max)
    sum_exp_max = np.sum(exp_k_distances_centered)
    # Add stability check for sum_exp_max, as seen in Inspirations 1, 2, 3
    if sum_exp_max == 0: return np.inf, np.zeros_like(x)
    smooth_max = (c_max + np.log(sum_exp_max)) / k
    
    # Manual Numba-compatible LogSumExp implementation for smooth_min
    scaled_distances_min = -k * distances
    c_min = np.max(scaled_distances_min)
    if not np.isfinite(c_min): return np.inf, np.zeros_like(x)
    exp_neg_k_distances_centered = np.exp(scaled_distances_min - c_min)
    sum_exp_min = np.sum(exp_neg_k_distances_centered)
    # Add stability check for sum_exp_min, as seen in Inspirations 1, 2, 3
    if sum_exp_min == 0: return np.inf, np.zeros_like(x)
    smooth_min = -(c_min + np.log(sum_exp_min)) / k

    if smooth_max < 1e-10:
        return np.inf, np.zeros_like(x) # Penalize if smooth_max is too small

    value = -smooth_min / smooth_max

    # Gradient Calculation
    grad_d_matrix = _distances_gradients_numba(points, n, d, distances, pdist_indices)
    
    weights_max = exp_k_distances_centered / sum_exp_max
    weights_min = exp_neg_k_distances_centered / sum_exp_min

    grad_S_max = np.dot(weights_max, grad_d_matrix)
    grad_S_min = np.dot(weights_min, grad_d_matrix)

    grad_F = -(grad_S_min * smooth_max - smooth_min * grad_S_max) / (smooth_max ** 2)

    return value, grad_F


def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    This implementation uses a hybrid optimization strategy:
    1. Global search with Differential Evolution on the exact objective function,
       initialized with a mix of diverse configurations (Sobol, perturbed grid, perturbed circle, random).
    2. Local refinement with L-BFGS-B on a smooth approximation of the objective function,
       using analytical gradients and a k-continuation scheme for robustness and precision.
    3. Final polishing step with a gradient-free optimizer on the exact objective.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Use global constants
    n = N_POINTS
    d = N_DIMS
    n_dims = N_PARAMS

    main_seed = 42 # Base seed for reproducibility
    bounds = [(0, 1)] * n_dims
    
    # Precompute pdist_indices once for efficiency in gradient calculation
    pdist_indices = _get_pdist_indices_cached(n)

    # --- Phase 1: Global Search with Multi-start Differential Evolution & Advanced Initialization ---
    # Implement multi-start Differential Evolution to escape poor local optima more effectively,
    # as mandated by the problem context for high-dimensional, multi-modal landscapes.
    num_de_runs = 5 # Number of independent DE runs
    
    # Reduced popsize_multiplier to allow for more DE runs within a reasonable total time,
    # and to reduce the computational cost per DE iteration.
    popsize_multiplier = 10 # Effective population size per DE generation: 10 * N_PARAMS = 10 * 32 = 320 individuals
    de_maxiter_per_run = 1500 # Reduced maxiter per run, total iterations across all runs: 5 * 1500 = 7500

    best_overall_ratio = -np.inf # Initialize with negative infinity for maximization
    best_overall_x = None

    for run_idx in range(num_de_runs):
        current_run_seed = main_seed + run_idx # Vary seed for each run for diversity
        rng = np.random.default_rng(current_run_seed) # New RNG for diverse initial populations
        
        de_population_size = popsize_multiplier * n_dims
        initial_population_list = []

        # Generate diverse initial populations for EACH DE run, ensuring different starting points
        num_strategies = 5
        num_inits_per_strategy = de_population_size // num_strategies
        perturbation_scale = 0.03 # Consistent perturbation scale

        # 1. Perturbed 4x4 grid configurations
        grid_side = int(np.sqrt(n))
        x_coords_grid = np.linspace(0.05, 0.95, grid_side); y_coords_grid = np.linspace(0.05, 0.95, grid_side)
        xv, yv = np.meshgrid(x_coords_grid, y_coords_grid)
        base_grid_config = np.vstack([xv.ravel(), yv.ravel()]).T.flatten()
        for _ in range(num_inits_per_strategy):
            perturbation = rng.uniform(-perturbation_scale, perturbation_scale, n_dims)
            initial_population_list.append(np.clip(base_grid_config + perturbation, 0, 1))

        # 2. Perturbed circular (16-gon) configurations
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        base_circle_config = np.vstack([0.5 + 0.49 * np.cos(theta), 0.5 + 0.49 * np.sin(theta)]).T.flatten()
        for _ in range(num_inits_per_strategy):
            perturbation = rng.uniform(-perturbation_scale, perturbation_scale, n_dims)
            initial_population_list.append(np.clip(base_circle_config + perturbation, 0, 1))

        # 3. Perturbed Hexagonal lattice
        hex_points_list = []
        num_rows_hex, num_cols_hex = 4, 4
        x_spacing = 0.8 / (num_cols_hex - 1 + 0.5); y_spacing = 0.8 * np.sqrt(3)/2 / (num_rows_hex - 1)
        for row_idx in range(num_rows_hex):
            y = row_idx * y_spacing
            x_offset = 0.5 * x_spacing if row_idx % 2 == 1 else 0.0
            for col_idx in range(num_cols_hex):
                hex_points_list.append([col_idx * x_spacing + x_offset, y])
        base_hex_points = np.array(hex_points_list)[:n]
        base_hex_points -= np.min(base_hex_points, axis=0)
        base_hex_points /= (np.max(base_hex_points, axis=0) - np.min(base_hex_points, axis=0) + 1e-12)
        base_hex_points = 0.1 + 0.8 * base_hex_points
        base_hex_coords_flat = base_hex_points.flatten()
        for _ in range(num_inits_per_strategy):
            perturbation = rng.uniform(-perturbation_scale, perturbation_scale, n_dims)
            initial_population_list.append(np.clip(base_hex_coords_flat + perturbation, 0, 1))

        # 4. Perturbed Sunflower Pattern
        golden_angle = np.pi * (3 - np.sqrt(5))
        sunflower_points_list = []
        for i in range(n):
            r_val = np.sqrt(i + 0.5) / np.sqrt(n - 0.5)
            theta_val = i * golden_angle
            x = r_val * np.cos(theta_val)
            y = r_val * np.sin(theta_val)
            sunflower_points_list.append([x, y])
        base_sunflower_points = np.array(sunflower_points_list)
        min_s_coords = np.min(base_sunflower_points, axis=0)
        max_s_coords = np.max(base_sunflower_points, axis=0)
        base_sunflower_points_scaled = (base_sunflower_points - min_s_coords) / (max_s_coords - min_s_coords + 1e-12)
        base_sunflower_points_scaled = 0.1 + 0.8 * base_sunflower_points_scaled
        base_sunflower_coords_flat = base_sunflower_points_scaled.flatten()
        for _ in range(num_inits_per_strategy):
            perturbation = rng.uniform(-perturbation_scale, perturbation_scale, n_dims)
            initial_population_list.append(np.clip(base_sunflower_coords_flat + perturbation, 0, 1))

        # 5. Quasi-Random (Sobol) initializations
        num_sobol_inits = de_population_size - len(initial_population_list)
        if num_sobol_inits > 0:
            # Use current_run_seed for Sobol sampler for consistency within the run
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=current_run_seed)
            initial_population_list.extend(sampler.random(num_sobol_inits))

        initial_de_population = np.array(initial_population_list)

        de_result = differential_evolution(
            func=_exact_objective_function_numba, # Use the Numba-accelerated true objective
            args=(n, d),
            bounds=bounds,
            seed=current_run_seed, # Use current_run_seed for reproducibility of this specific run
            popsize=popsize_multiplier, 
            maxiter=de_maxiter_per_run, # Use reduced maxiter per run
            init=initial_de_population,
            strategy='rand1bin',
            tol=1e-9,
            mutation=(0.7, 1.0),
            recombination=0.9,
            workers=-1,
            disp=False,
            polish=False
        )
        
        # Check if this DE run found a better solution (remember objective is -dmin/dmax)
        current_de_ratio = -de_result.fun
        if current_de_ratio > best_overall_ratio:
            best_overall_ratio = current_de_ratio
            best_overall_x = de_result.x
    
    # After all DE runs, use the best solution found for local refinement
    global_opt_x = best_overall_x
    
    # --- Phase 2: Local Refinement with L-BFGS-B using k-continuation and Analytical Gradients ---
    # pdist_indices is already cached and global.
    current_x = global_opt_x
    
    # Extended k-sequence for even higher precision in smooth approximation, as in Inspiration 2
    k_values = [100.0, 500.0, 2500.0, 10000.0, 50000.0, 250000.0, 1000000.0, 2500000.0, 5000000.0, 10000000.0] 
    
    for k_val in k_values:
        refinement_result = minimize(
            _smooth_objective_and_grad_numba,
            current_x,
            args=(n, d, k_val, pdist_indices),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'disp': False, 'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-11} # Increased maxiter, stricter tolerances as in Inspiration 2
        )
        current_x = refinement_result.x

    # --- Phase 3: Final Polish with a Gradient-Free Method ---
    # Use Nelder-Mead on the exact objective function for a final, high-precision polish.
    final_polish_result = minimize(
        lambda x, n_arg, d_arg: _exact_objective_function_numba(x, n_arg, d_arg),
        current_x,
        args=(n, d),
        method='Nelder-Mead',
        options={'disp': False, 'maxiter': 7000, 'xatol': 1e-11, 'fatol': 1e-11} # Adjusted maxiter and stricter tolerances as in Inspiration 2
    )

    optimized_flat_points = final_polish_result.x
    
    points = optimized_flat_points.reshape(n, d)

    return points
# EVOLVE-BLOCK-END