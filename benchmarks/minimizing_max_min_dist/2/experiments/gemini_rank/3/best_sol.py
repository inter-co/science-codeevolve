# EVOLVE-BLOCK-START
import numpy as np 
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc # Added for Sobol sequences and Latin Hypercube Sampling
from scipy.special import logsumexp # For numerically stable smooth objective

# Helper to calculate the true min/max ratio (for evaluation and DE global search)
def _calculate_min_max_ratio(points_arr: np.ndarray) -> float:
    distances = pdist(points_arr)
    if distances.size == 0:
        return 0.0
    dmin, dmax = np.min(distances), np.max(distances)
    return dmin / dmax if dmax > 1e-9 else 0.0 # Use a small threshold to avoid division by zero

# The true objective function for Differential Evolution (non-smooth)
def _true_objective(flat_points: np.ndarray, n: int, d: int) -> float:
    """Objective function for DE: negative of min/max distance ratio."""
    return -_calculate_min_max_ratio(flat_points.reshape(n, d))

def _log_sum_exp_objective(x: np.ndarray, n_points: int, n_dims: int, p: float) -> float:
    """
    A smooth, differentiable surrogate for the min/max ratio objective.
    Minimizing this function is approximately equivalent to maximizing d_min / d_max.
    It minimizes log(d_max / d_min).
    """
    points = x.reshape(n_points, n_dims)
    distances = pdist(points)
    
    epsilon = 1e-12
    distances = np.maximum(distances, epsilon)
    log_distances = np.log(distances) # Convert to log-distances for logsumexp

    # log-sum-exp for d_max: log(sum(exp(p * log(d_ij)))) / p ≈ log(d_max)
    term_pos = logsumexp(p * log_distances)
    
    # log-sum-exp for d_min: log(sum(exp(-p * log(d_ij)))) / p ≈ log(d_min)
    term_neg = logsumexp(-p * log_distances)
    
    return (term_pos + term_neg) / p

def _gradient_log_sum_exp_objective(x: np.ndarray, n_points: int, n_dims: int, p: float) -> np.ndarray:
    """
    Analytical gradient for the smooth log-sum-exp objective.
    Providing an exact gradient significantly improves L-BFGS-B performance.
    (Adopted from Inspiration Program 1)
    """
    points = x.reshape(n_points, n_dims)
    
    distances_condensed = pdist(points)
    epsilon = 1e-12
    distances_condensed = np.maximum(distances_condensed, epsilon)
    log_distances_condensed = np.log(distances_condensed)
    
    L_condensed = p * log_distances_condensed
    S_pos = logsumexp(L_condensed)
    S_neg = logsumexp(-L_condensed)

    # Weights for gradient contributions
    weights = (np.exp(L_condensed - S_pos) - np.exp(-L_condensed - S_neg)) / (distances_condensed**2)

    grad = np.zeros_like(points)
    
    k = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            diff = points[i] - points[j]
            grad_contribution = weights[k] * diff
            grad[i] += grad_contribution
            grad[j] -= grad_contribution
            k += 1
            
    return grad.flatten()

def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalizes a point configuration to fit maximally within the [0,1]x[0,1] square."""
    if points is None or points.size == 0:
        return np.array([])
    if points.shape[0] < 2:
        return np.full_like(points, 0.5)

    points_copy = points.copy()
    points_copy -= points_copy.min(axis=0) # Translate to origin
    
    max_extent = points_copy.max()
    
    if max_extent < 1e-9: # If all points are identical, center them to (0.5, 0.5)
        return np.full_like(points, 0.5)
        
    points_copy /= max_extent # Scale to fit within [0,1]
    return points_copy # No clip needed as scaling should put points within [0,1]

def _generate_diverse_initial_populations(n_points: int, n_dims: int, popsize: int, jitter_scale: float = 0.08, seed: int = None) -> np.ndarray:
    """
    Generates a diverse initial population for DE using a mix of strategies, inspired by
    the best-performing programs. Includes perturbed grid, circle, and hexagonal patterns,
    plus Sobol and Latin Hypercube Sampling (LHS) for robust space-filling.
    (Adapted from Inspiration Program 1)
    """
    rng = np.random.default_rng(seed)
    initial_population = np.empty((popsize, n_points * n_dims))
    current_idx = 0

    # Robustly divide population size among 5 strategies
    if popsize < 5:
        counts = [0] * 5
        for i in range(popsize): counts[i] = 1
        num_grid, num_circle, num_hex, num_sobol, num_lhs = counts
    else:
        base_count = popsize // 5
        remainder = popsize % 5
        counts = [base_count] * 5
        for i in range(remainder): counts[i] += 1
        num_grid, num_circle, num_hex, num_sobol, num_lhs = counts

    # --- Strategy 1: Perturbed 4x4 Grid ---
    if num_grid > 0 and n_points == 16 and n_dims == 2:
        n_side = int(np.sqrt(n_points))
        grid_steps = np.linspace(0, 1, n_side + 1)
        cell_centers = (grid_steps[:-1] + grid_steps[1:]) / 2
        xs, ys = np.meshgrid(cell_centers, cell_centers)
        base_grid = np.vstack([xs.ravel(), ys.ravel()]).T
        for i in range(num_grid):
            jitter = (rng.random((n_points, n_dims)) - 0.5) * 2 * jitter_scale
            initial_population[current_idx + i] = np.clip(base_grid + jitter, 0, 1).flatten()
        current_idx += num_grid

    # --- Strategy 2: Perturbed 16-gon Circle ---
    if num_circle > 0 and n_points == 16 and n_dims == 2:
        center = np.array([0.5, 0.5])
        radius = 0.5 - (jitter_scale * 1.5)
        if radius < 0.1: radius = 0.1 # Ensure radius is not too small
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        base_circle = np.array([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)]).T
        for i in range(num_circle):
            jitter = (rng.random((n_points, n_dims)) - 0.5) * 2 * jitter_scale
            initial_population[current_idx + i] = np.clip(base_circle + jitter, 0, 1).flatten()
        current_idx += num_circle
        
    # --- Strategy 3: Perturbed Hexagonal Lattice ---
    if num_hex > 0 and n_points == 16 and n_dims == 2:
        hex_rows, hex_cols = 4, 4 
        hex_base_points = np.array([[i + (j % 2) * 0.5, j * np.sqrt(3) / 2] for j in range(hex_rows) for i in range(hex_cols)])
        normalized_hex = _normalize_points(hex_base_points)
        for i in range(num_hex):
            jitter = (rng.random((n_points, n_dims)) - 0.5) * 2 * jitter_scale
            initial_population[current_idx + i] = np.clip(normalized_hex + jitter, 0, 1).flatten()
        current_idx += num_hex

    # --- Strategy 4: Sobol Quasi-random sequence ---
    if num_sobol > 0:
        sampler = qmc.Sobol(d=n_points * n_dims, seed=rng.integers(0, 2**30))
        initial_population[current_idx : current_idx + num_sobol] = sampler.random(n=num_sobol)
        current_idx += num_sobol

    # --- Strategy 5: Latin Hypercube Sampling (LHS) ---
    if num_lhs > 0:
        sampler = qmc.LatinHypercube(d=n_points * n_dims, seed=rng.integers(0, 2**30))
        initial_population[current_idx : current_idx + num_lhs] = sampler.random(n=num_lhs)
        current_idx += num_lhs

    # --- Fill any remaining slots with random uniform (e.g., if geometric patterns failed) ---
    if current_idx < popsize:
        num_remaining = popsize - current_idx
        initial_population[current_idx:] = rng.random((num_remaining, n_points * n_dims))

    return initial_population

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2D to maximize the min/max distance ratio using a hybrid strategy:
    1. Multi-start Differential Evolution (DE) with an enriched, dynamically randomized initial population.
    2. Local refinement of the best DE result using L-BFGS-B on a smooth surrogate objective.
    3. Final local refinement using Nelder-Mead on the true (non-smooth) objective for polishing.
    4. Final normalization of the point set.
    """
    n_points = 16
    n_dims = 2
    n_coords = n_points * n_dims
    bounds = [(0.0, 1.0)] * n_coords

    best_score = np.inf
    best_points_found = None

    base_seed = 42
    master_rng = np.random.default_rng(base_seed) # Master RNG for overall reproducibility

    # Hyperparameters (tuned for potentially higher min_max_ratio)
    n_starts = 10 # Number of multi-start DE runs (kept at 10 to manage runtime, but could be increased for more thorough search)
    de_popsize = 100 # Population size for differential_evolution
    de_maxiter = 5000 # Further increased max iterations for deeper global search
    p_smooth = 2000.0 # Significantly increased power for a sharper, more accurate smooth log-sum-exp objective
    lbfgsb_maxiter = 3000 # Further increased max iterations for L-BFGS-B refinement
    
    # Nelder-Mead options for final polishing
    nm_maxiter = 1200 # Further increased max iterations for final polishing
    nm_fatol = 1e-8
    nm_xatol = 1e-8

    for i in range(n_starts):
        current_seed = master_rng.integers(0, 2**32 - 1)
        
        # --- Enriched Initial Population Generation (using helper from inspiration) ---
        # Reduced jitter_scale to preserve initial structured patterns (grid/circle) more effectively.
        initial_population_array = _generate_diverse_initial_populations(
            n_points, n_dims, de_popsize, jitter_scale=0.03, seed=current_seed
        )

        # --- Stage 1: Global search with Differential Evolution (True Objective) ---
        de_result = differential_evolution(
            func=_true_objective, # Use the true (non-smooth) objective
            bounds=bounds, args=(n_points, n_dims),
            strategy='currenttobest1bin', 
            maxiter=de_maxiter, popsize=de_popsize, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.9,
            disp=False, polish=False, workers=-1,
            seed=current_seed, init=initial_population_array
        )

        # Update best score if DE found a better solution
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_points_found = de_result.x

    # --- Stage 2: Local Refinement with L-BFGS-B (Smooth Surrogate Objective with Analytical Gradient) ---
    if best_points_found is not None:
        lbfgsb_result = minimize(
            fun=_log_sum_exp_objective,
            x0=best_points_found,
            args=(n_points, n_dims, p_smooth),
            method='L-BFGS-B',
            jac=_gradient_log_sum_exp_objective, # Provide the analytical gradient for speed and accuracy
            bounds=bounds,
            options={'maxiter': lbfgsb_maxiter, 'ftol': 1e-10, 'gtol': 1e-9}
        )
        
        # --- Stage 3: Final Polish with Nelder-Mead on True Objective ---
        final_polish_result = minimize(
            fun=_true_objective, # Use the true (non-smooth) objective
            x0=lbfgsb_result.x, # Start from L-BFGS-B's best solution
            args=(n_points, n_dims),
            method='Nelder-Mead',
            bounds=bounds, 
            options={'maxiter': nm_maxiter, 'fatol': nm_fatol, 'xatol': nm_xatol, 'disp': False}
        )

        # Safeguard: Only accept the fully refined result if it's truly better on the original objective
        # This prevents the smooth surrogate from occasionally leading to a worse true score.
        if final_polish_result.fun < best_score:
            best_points_found = final_polish_result.x

    # Fallback if no solution was found by DE or refinement
    if best_points_found is None:
        random_seed_fallback = master_rng.integers(0, 2**32 - 1) 
        fallback_rng = np.random.default_rng(seed=random_seed_fallback)
        best_points_found = fallback_rng.uniform(0, 1, size=n_coords)

    final_points = best_points_found.reshape((n_points, n_dims))
    
    # --- Final Normalization ---
    return _normalize_points(final_points)
# EVOLVE-BLOCK-END