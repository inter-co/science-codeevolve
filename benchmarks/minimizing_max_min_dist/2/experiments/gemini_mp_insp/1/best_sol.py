# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
from scipy.stats import qmc # Added for Sobol sequence initialization
import numba
from numba import float64, int64, prange # For Numba type hints and parallel loop

# Global constants for the problem
N_POINTS = 16
DIMENSIONS = 2
N_PARAMS = N_POINTS * DIMENSIONS # Total number of parameters (16 points * 2 dimensions = 32)

# --- Numba-optimized core functions (Inspired by Inspiration Programs 1, 2 & 3) ---

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

_pdist_indices_cache = None
def _get_pdist_indices_cached(n: int) -> np.ndarray:
    """
    Helper to get pdist indices once for efficiency in gradient calculation.
    """
    global _pdist_indices_cache
    num_distances = n * (n - 1) // 2
    if _pdist_indices_cache is None or _pdist_indices_cache.shape[0] != num_distances:
        indices = np.empty((num_distances, 2), dtype=np.int64)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                indices[k, 0] = i
                indices[k, 1] = j
                k += 1
        _pdist_indices_cache = indices
    return _pdist_indices_cache

@numba.jit(nopython=True, fastmath=True)
def _distances_gradients_numba(points: np.ndarray, n: int, d: int, distances: np.ndarray, pdist_indices: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of each pairwise distance with respect to each point coordinate.
    Optimized with Numba and vectorized difference calculation.
    """
    num_distances = len(distances)
    grad_d_matrix = np.zeros((num_distances, n * d), dtype=np.float64)
    # Vectorized calculation of differences (p1 - p2) for all pairs
    diffs = points[pdist_indices[:, 0]] - points[pdist_indices[:, 1]]
    
    for i in range(num_distances):
        # Add a small epsilon for stability if points collapse
        dist_inv = 1.0 / (distances[i] + 1e-12)
        
        for dim in range(d):
            grad_val = diffs[i, dim] * dist_inv
            # Gradient for point 1
            grad_d_matrix[i, pdist_indices[i, 0] * d + dim] = grad_val
            # Gradient for point 2
            grad_d_matrix[i, pdist_indices[i, 1] * d + dim] = -grad_val
            
    return grad_d_matrix

@numba.jit(nopython=True, fastmath=True)
def _exact_objective_function_numba(x_flat: np.ndarray, n: int, d: int) -> float:
    """
    Calculates the negative of the exact min_max_ratio.
    Optimized with Numba for performance.
    """
    points = x_flat.reshape(n, d)
    distances = _pairwise_distances_numba(points, n, d)
    
    if len(distances) == 0:
        return np.inf # Penalty for invalid configurations

    dmin = np.min(distances)
    dmax = np.max(distances)
    
    # Penalize points collapsing or being too close
    if dmin < 1e-9: # Tighter tolerance from inspirations
        return np.inf # Extremely high penalty
        
    # Penalize cases where dmax is effectively zero (all points at same spot)
    if dmax < 1e-9:
        return np.inf # Extremely high penalty

    return -dmin / dmax # Minimize the negative ratio

# The _smooth_objective_value_numba function is removed as Differential Evolution
# will now operate directly on the _exact_objective_function_numba, aligning
# with the most successful inspiration programs for better global search.

@numba.jit(nopython=True, fastmath=True)
def _smooth_objective_and_grad_numba(x_flat: np.ndarray, n: int, d: int, k: float, pdist_indices: np.ndarray):
    """
    Calculates the negative of the ratio of a smooth minimum distance to a
    smooth maximum distance using Numba-optimized LogSumExp approximations,
    and its analytical gradient. (Inspired by Inspiration Program 2)
    """
    points = x_flat.reshape(n, d)
    distances = _pairwise_distances_numba(points, n, d)

    if np.min(distances) < 1e-10: # Tighter tolerance from inspirations
        return np.inf, np.zeros_like(x_flat) # Penalty and zero gradient

    # --- Value Calculation (Numerically Stable LogSumExp) ---
    # For smooth_max
    scaled_distances_max = k * distances
    c_max = np.max(scaled_distances_max)
    if not np.isfinite(c_max): return np.inf, np.zeros_like(x_flat)
    
    exp_k_distances_centered = np.exp(scaled_distances_max - c_max)
    sum_exp_max = np.sum(exp_k_distances_centered)
    
    if sum_exp_max == 0: return np.inf, np.zeros_like(x_flat)
    
    smooth_max = (c_max + np.log(sum_exp_max)) / k
    
    # For smooth_min
    scaled_distances_min = -k * distances
    c_min = np.max(scaled_distances_min)
    if not np.isfinite(c_min): return np.inf, np.zeros_like(x_flat)
    
    exp_neg_k_distances_centered = np.exp(scaled_distances_min - c_min)
    sum_exp_min = np.sum(exp_neg_k_distances_centered)
    
    if sum_exp_min == 0: return np.inf, np.zeros_like(x_flat)
    
    smooth_min = -(c_min + np.log(sum_exp_min)) / k

    if smooth_max < 1e-10: # Tighter tolerance
        return np.inf, np.zeros_like(x_flat) # Large penalty
        
    value = -smooth_min / smooth_max
    
    if not np.isfinite(value):
        return value, np.zeros_like(x_flat)

    # --- Gradient Calculation ---
    grad_d_matrix = _distances_gradients_numba(points, n, d, distances, pdist_indices)

    weights_max = exp_k_distances_centered / sum_exp_max
    weights_min = exp_neg_k_distances_centered / sum_exp_min

    grad_smooth_max = np.dot(weights_max, grad_d_matrix)
    grad_smooth_min = np.dot(weights_min, grad_d_matrix)

    grad_F = -(grad_smooth_min * smooth_max - smooth_min * grad_smooth_max) / (smooth_max**2)
    return value, grad_F

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2D to maximize the min/max distance ratio using a three-phase hybrid optimization:
    1. Global Search: Differential Evolution on a SMOOTH objective with a diverse initial population.
    2. Local Refinement: L-BFGS-B with analytical gradients and k-continuation on the smooth objective.
    3. Final Polish: Nelder-Mead on the exact objective for high-precision final tuning.
    This strategy is inspired by the most successful inspiration programs.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Set a seed for reproducibility.
    main_seed = 42
    rng = np.random.default_rng(seed=main_seed) # For custom initial population generation

    # Define the bounds for each coordinate to keep points within the unit square [0,1]x[0,1]
    bounds = [(0, 1)] * N_PARAMS

    # Precompute pdist_indices once for efficiency in gradient calculation
    pdist_indices = _get_pdist_indices_cached(N_POINTS)
    
    # --- Custom Initial Population Strategy for Differential Evolution ---
    # Generate a diverse initial population to improve exploration.
    popsize_multiplier = 20 # Increased multiplier for a more thorough search
    de_population_size = popsize_multiplier * N_PARAMS # 20 * 32 = 640 individuals
    initial_population_list = []

    # Strategy 1: Perturbed 4x4 Grid (20% of population)
    num_inits_per_strategy = de_population_size // 5 # Rebalanced for 5 strategies
    grid_side = int(np.sqrt(N_POINTS))
    x_coords_grid = np.linspace(0.1, 0.9, grid_side); y_coords_grid = np.linspace(0.1, 0.9, grid_side)
    xv, yv = np.meshgrid(x_coords_grid, y_coords_grid)
    base_grid_coords_flat = np.vstack([xv.ravel(), yv.ravel()]).T.flatten()

    perturbation_scale = 0.03
    for _ in range(num_inits_per_strategy):
        perturbation = rng.uniform(-perturbation_scale, perturbation_scale, size=N_PARAMS)
        initial_population_list.append(np.clip(base_grid_coords_flat + perturbation, 0, 1))

    # Strategy 2: Perturbed Regular 16-gon (20% of population)
    center = np.array([0.5, 0.5]); radius = 0.45
    angles = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    base_polygon_coords_flat = np.array([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]).T.flatten()

    for _ in range(num_inits_per_strategy):
        perturbation = rng.uniform(-perturbation_scale, perturbation_scale, size=N_PARAMS)
        initial_population_list.append(np.clip(base_polygon_coords_flat + perturbation, 0, 1))

    # Strategy 3: Perturbed Hexagonal lattice (20% of population)
    hex_points_list = []
    num_rows_hex, num_cols_hex = 4, 4
    x_spacing = 0.8 / (num_cols_hex - 1 + 0.5); y_spacing = 0.8 * np.sqrt(3)/2 / (num_rows_hex - 1)
    for row_idx in range(num_rows_hex):
        y = row_idx * y_spacing
        x_offset = 0.5 * x_spacing if row_idx % 2 == 1 else 0.0
        for col_idx in range(num_cols_hex):
            hex_points_list.append([col_idx * x_spacing + x_offset, y])
    base_hex_points = np.array(hex_points_list)[:N_POINTS]
    base_hex_points -= np.min(base_hex_points, axis=0)
    base_hex_points /= np.max(base_hex_points) if np.max(base_hex_points) > 1e-12 else 1.0
    base_hex_points = 0.1 + 0.8 * base_hex_points
    base_hex_coords_flat = base_hex_points.flatten()

    for _ in range(num_inits_per_strategy):
        perturbation = rng.uniform(-perturbation_scale, perturbation_scale, size=N_PARAMS)
        initial_population_list.append(np.clip(base_hex_coords_flat + perturbation, 0, 1))

    # Strategy 4: Perturbed Sunflower Pattern (20% of population) - New strategy from inspirations
    golden_angle = np.pi * (3 - np.sqrt(5))
    sunflower_points_list = []
    for i in range(N_POINTS):
        r_val = np.sqrt(i + 0.5) / np.sqrt(N_POINTS - 0.5) # Scale radius
        theta_val = i * golden_angle
        x = r_val * np.cos(theta_val)
        y = r_val * np.sin(theta_val)
        sunflower_points_list.append([x, y])
    base_sunflower_points = np.array(sunflower_points_list)
    # Scale and center sunflower points to fit within [0.1, 0.9]
    min_s_coords = np.min(base_sunflower_points, axis=0)
    max_s_coords = np.max(base_sunflower_points, axis=0)
    # Add small epsilon to denominator to prevent division by zero if all points are identical
    base_sunflower_points_scaled = (base_sunflower_points - min_s_coords) / (max_s_coords - min_s_coords + 1e-12)
    base_sunflower_points_scaled = 0.1 + 0.8 * base_sunflower_points_scaled # Scale to [0.1, 0.9]
    base_sunflower_coords_flat = base_sunflower_points_scaled.flatten()

    for _ in range(num_inits_per_strategy):
        perturbation = rng.uniform(-perturbation_scale, perturbation_scale, size=N_PARAMS)
        initial_population_list.append(np.clip(base_sunflower_coords_flat + perturbation, 0, 1))

    # Strategy 5: Quasi-Random (Sobol) initializations for the remaining slots (20% of population)
    num_sobol_inits = de_population_size - len(initial_population_list)
    if num_sobol_inits > 0:
        sobol_sampler = qmc.Sobol(d=N_PARAMS, scramble=True, seed=main_seed)
        initial_population_list.extend(sobol_sampler.random(num_sobol_inits))

    initial_population_array = np.array(initial_population_list)
    # --- End of Smart Initialization Strategy ---

    # --- Step 1: Global optimization with Differential Evolution on the EXACT objective ---
    # Aligned with successful inspiration programs for robust global search.
    de_result = differential_evolution(
        _exact_objective_function_numba, # Use Numba-optimized EXACT objective function
        bounds=bounds,
        args=(N_POINTS, DIMENSIONS),
        strategy='rand1bin', # Changed to 'rand1bin' for more exploration (from Insp 1 & 3)
        maxiter=6000, # Increased iterations for more thorough global search on harder landscape
        popsize=popsize_multiplier,
        tol=1e-8, # Stricter tolerance for DE convergence
        mutation=(0.5, 1.2), # More explorative mutation
        recombination=0.8,
        seed=main_seed,
        disp=False,
        workers=-1, # Use all available CPU cores for parallel evaluation
        init=initial_population_array,
        polish=False # Rely on separate L-BFGS-B for detailed polishing
    )
    
    # --- Step 2: Local refinement with L-BFGS-B using k-continuation ---
    # Start from the best result found by Differential Evolution.
    current_x = de_result.x
    
    # Sequence of k values for the continuation method.
    # Starting from a moderate k and ramping up aggressively to very high values
    # for ultimate precision in approximating the min/max function (inspired by Inspiration 2 & 3).
    k_values = [100.0, 500.0, 2500.0, 10000.0, 50000.0, 250000.0, 1000000.0, 2500000.0, 5000000.0, 10000000.0, 25000000.0] # Extended k-values for highest precision
    
    for k_val in k_values:
        lbfgsb_result = minimize(
            _smooth_objective_and_grad_numba, # Use the smooth objective with analytical gradients
            current_x,
            args=(N_POINTS, DIMENSIONS, k_val, pdist_indices), # Pass pdist_indices for gradient calculation
            method='L-BFGS-B',
            jac=True, # Explicitly tell minimize that gradient is provided
            bounds=bounds,
            options={'disp': False, 'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-12} # Increased maxiter and stricter gtol for precise local search
        )
        current_x = lbfgsb_result.x
    
    # --- Step 3: Final Polish with Nelder-Mead on the EXACT objective ---
    # This step snaps the solution to a true optimum of the non-smooth problem.
    final_polish_result = minimize(
        _exact_objective_function_numba,
        current_x,
        args=(N_POINTS, DIMENSIONS),
        method='Nelder-Mead',
        options={'disp': False, 'maxiter': 7000, 'xatol': 1e-11, 'fatol': 1e-11} # Increased maxiter for thorough final polish
    )
    
    points = final_polish_result.x.reshape(N_POINTS, DIMENSIONS)

    return points
# EVOLVE-BLOCK-END