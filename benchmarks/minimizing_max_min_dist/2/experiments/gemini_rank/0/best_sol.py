# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize # Import minimize for local refinement
from scipy.spatial.distance import pdist
from scipy.special import logsumexp # For numerically stable smooth objective
from scipy.stats import qmc # For Sobol/LHS sequences

# --- Global Constants ---
N_POINTS = 16
DIMS = 2

# Helper to calculate the true min/max ratio (for evaluation and DE global search)
def _calculate_min_max_ratio(points_arr: np.ndarray) -> float:
    distances = pdist(points_arr)
    if distances.size == 0:
        return 0.0
    dmin, dmax = np.min(distances), np.max(distances)
    return dmin / dmax if dmax > 1e-9 else 0.0 # Use a small threshold to avoid division by zero

# The true objective function for Differential Evolution and SLSQP (non-smooth)
# CRITICAL IMPROVEMENT from Inspiration 3: Normalizes points *within* the objective.
def _true_objective(flat_points: np.ndarray) -> float:
    """
    Objective function for DE and SLSQP: negative of min/max distance ratio
    of a configuration *normalized to the unit square*.
    This forces the optimizer to consider how well the configuration fills the space,
    not just its scale-invariant ratio.
    """
    points = flat_points.reshape(N_POINTS, DIMS)
    normalized_points = _normalize_points(points) # Normalize points at each evaluation
    return -_calculate_min_max_ratio(normalized_points)

# Smooth surrogate objective function using log-sum-exp approximation for min/max.
# Uses global N_POINTS, DIMS (from Inspiration 3).
def _log_sum_exp_objective(x: np.ndarray, p: float) -> float:
    """
    A smooth, differentiable surrogate for the min/max ratio objective.
    Minimizing this function is approximately equivalent to maximizing d_min / d_max.
    It minimizes log(d_max / d_min).
    Args:
        x: Flattened 1D array of point coordinates.
        p: The power used in the log-sum-exp approximation.
    Returns:
        The score to be minimized.
    """
    points = x.reshape(N_POINTS, DIMS)
    
    distances = pdist(points)
    
    epsilon = 1e-12
    distances = np.maximum(distances, epsilon)
    log_distances = np.log(distances)

    term_pos = logsumexp(p * log_distances)
    term_neg = logsumexp(-p * log_distances)
    
    return (term_pos + term_neg) / p

# Analytical gradient for the smooth log-sum-exp objective.
# Uses global N_POINTS, DIMS (from Inspiration 3).
def _gradient_log_sum_exp_objective(x: np.ndarray, p: float) -> np.ndarray:
    """
    Analytical gradient for the smooth log-sum-exp objective.
    Providing an exact gradient significantly improves L-BFGS-B performance.
    """
    points = x.reshape(N_POINTS, DIMS)
    
    distances_condensed = pdist(points)
    epsilon = 1e-12
    distances_condensed = np.maximum(distances_condensed, epsilon)
    log_distances_condensed = np.log(distances_condensed)
    
    L_condensed = p * log_distances_condensed
    S_pos = logsumexp(L_condensed)
    S_neg = logsumexp(-L_condensed)

    weights = (np.exp(L_condensed - S_pos) - np.exp(-L_condensed - S_neg)) / (distances_condensed**2)

    grad = np.zeros_like(points)
    
    k = 0
    for i in range(N_POINTS):
        for j in range(i + 1, N_POINTS):
            diff = points[i] - points[j]
            grad_contribution = weights[k] * diff
            grad[i] += grad_contribution
            grad[j] -= grad_contribution
            k += 1
            
    return grad.flatten()


# Helper function for initial population generation based on diverse strategies.
# Uses global N_POINTS, DIMS and enhanced strategy (from Inspiration 3).
def _generate_diverse_initial_population(popsize: int, jitter_scale: float = 0.08, seed: int = None) -> np.ndarray:
    """
    Generates a diverse initial population for Differential Evolution using a mix of strategies:
    unperturbed/perturbed grid, hexagonal, circle (16-gon), Sobol sequences, and random points.
    """
    rng = np.random.default_rng(seed)
    initial_population = np.empty((popsize, N_POINTS * DIMS))
    initial_population_list = []

    # Generator for unperturbed 4x4 Grid
    def unperturbed_grid_gen():
        grid_steps = np.linspace(0, 1, int(np.sqrt(N_POINTS)) + 1)
        cell_centers = (grid_steps[:-1] + grid_steps[1:]) / 2
        xs, ys = np.meshgrid(cell_centers, cell_centers)
        return np.vstack([xs.ravel(), ys.ravel()]).T.flatten()

    # Generator for unperturbed Hexagonal lattice
    def unperturbed_hex_gen():
        hex_rows, hex_cols = 4, 4 
        hex_base_points = np.array([[i + (j % 2) * 0.5, j * np.sqrt(3) / 2] for j in range(hex_rows) for i in range(hex_cols)])
        return _normalize_points(hex_base_points).flatten()

    # Generator for unperturbed 16-gon
    def unperturbed_poly_gen():
        center = np.array([0.5, 0.5])
        radius = 0.49 
        theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
        poly_points = np.array([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)]).T
        return poly_points.flatten()

    # Generator for perturbed versions of a base configuration
    def perturbed_gen(base_config_flat: np.ndarray):
        base_config_reshaped = base_config_flat.reshape(N_POINTS, DIMS)
        while True:
            jitter = (rng.random((N_POINTS, DIMS)) - 0.5) * jitter_scale * 2
            yield np.clip(base_config_reshaped + jitter, 0, 1).flatten()

    # Generator for Sobol sequences
    def sobol_gen():
        sampler = qmc.Sobol(d=N_POINTS * DIMS, seed=rng.integers(0, 2**30))
        for sample in sampler.random(n=popsize * 2): # Generate more to ensure enough unique samples
            yield sample

    # Generator for purely random points
    def random_gen():
        while True:
            yield rng.uniform(0, 1, size=(N_POINTS * DIMS))

    # Add unperturbed strong candidates
    if N_POINTS == 16 and DIMS == 2:
        initial_population_list.append(unperturbed_grid_gen())
        initial_population_list.append(unperturbed_hex_gen())
        initial_population_list.append(unperturbed_poly_gen())
    
    # Store base configurations for perturbation
    base_configs_flat = list(initial_population_list)

    # Initialize a list of generators to cycle through for the rest of the population
    active_generators = []
    if N_POINTS == 16 and DIMS == 2:
        for base_config_flat in base_configs_flat:
            active_generators.append(perturbed_gen(base_config_flat))
    
    active_generators.append(sobol_gen())
    active_generators.append(random_gen()) # Fallback for any remaining slots

    gen_idx = 0
    while len(initial_population_list) < popsize:
        initial_population_list.append(next(active_generators[gen_idx % len(active_generators)]))
        gen_idx += 1

    return np.array(initial_population_list[:popsize])


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalizes a point configuration to fit maximally within the [0,1]x[0,1] square."""
    if points is None or points.size == 0:
        return np.array([])
    
    # Translate points so the minimum of each axis is 0
    min_coords = points.min(axis=0)
    points_translated = points - min_coords
    
    # Find the maximum extent in any dimension
    max_coords = points.max(axis=0)
    range_coords = max_coords - min_coords
    
    max_overall_range = 0.0
    if np.any(range_coords > 1e-9): # Robust check for non-zero range
        max_overall_range = np.max(range_coords[range_coords > 1e-9])
    
    if max_overall_range > 1e-9:
        # Scale points to fit within the unit square, preserving aspect ratio
        points_normalized = points_translated / max_overall_range
    else:
        # If all points are identical, center them to [0.5, 0.5]
        points_normalized = np.full_like(points, 0.5)
    return points_normalized

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    This is achieved using a multi-stage optimization process:
    1. Multi-start Differential Evolution with diverse initializations and the true (non-smooth) objective.
    2. Local refinement using L-BFGS-B on a numerically stable smooth surrogate objective (with analytical gradient).
    3. Final local refinement using SLSQP on the true (non-smooth) objective, respecting bounds.
    4. Final normalization of the point set to maximally fill the unit square.

    Returns:
        points: np.ndarray of shape (16, 2) containing the (x, y) coordinates of the 16 points.
    """
    # Use global constants
    bounds = [(0, 1)] * (N_POINTS * DIMS)

    best_ratio = -np.inf
    best_points = None
    
    n_starts = 10
    base_seed = 42
    
    p_smooth = 800.0
    
    de_maxiter = 3000
    de_popsize = 100
    de_tol = 1e-5
    de_workers = -1
    
    for i in range(n_starts):
        current_seed = base_seed + i
        
        initial_population_for_run = _generate_diverse_initial_population(
            popsize=de_popsize, jitter_scale=0.08, seed=current_seed
        )
        
        # --- Stage 1: Global Search with Differential Evolution ---
        de_result = differential_evolution(
            _true_objective, # Uses the new _true_objective that normalizes internally
            bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=de_tol,
            seed=current_seed,
            disp=False,
            workers=de_workers,
            init=initial_population_for_run,
            strategy='currenttobest1bin',
            polish=False
        )

        # --- Stage 2: Local Refinement with L-BFGS-B on Smooth Surrogate (with analytical gradient) ---
        lbfgs_result = minimize(
            _log_sum_exp_objective,
            de_result.x,
            args=(p_smooth,), # Updated args for _log_sum_exp_objective
            method='L-BFGS-B',
            jac=_gradient_log_sum_exp_objective,
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        
        # --- Stage 3: Final Polish with SLSQP on True Objective ---
        final_result = minimize(
            _true_objective, # Uses the new _true_objective that normalizes internally
            lbfgs_result.x,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        # The points from final_result.x are already "normalized" by the _true_objective
        # if the optimization was successful. We still need to reshape and apply a final
        # normalization to ensure the output is canonical.
        raw_points = final_result.x.reshape(N_POINTS, DIMS)

        # Recalculate ratio for the final points to ensure accuracy, using the normalized version
        current_ratio = _calculate_min_max_ratio(_normalize_points(raw_points))
        
        # Update overall best solution
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = raw_points # Store the raw points, will normalize at the very end

    # Final normalization of the best found configuration to guarantee it spans the unit square.
    return _normalize_points(best_points)
# EVOLVE-BLOCK-END