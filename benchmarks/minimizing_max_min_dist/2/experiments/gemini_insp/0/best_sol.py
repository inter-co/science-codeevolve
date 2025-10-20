# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc

# Core objective function, renamed to _evaluate_points for clarity, based on IP1.
def _evaluate_points(flat_points_array: np.ndarray, n: int, d: int) -> float:
    """
    Evaluates the min/max distance ratio for a given set of points.
    The optimizer minimizes, so we return the negative ratio.
    """
    points = flat_points_array.reshape((n, d))
    distances = pdist(points)

    if len(distances) == 0:
        return np.inf

    dmin = np.min(distances)
    
    if dmin < 1e-9:
        return np.inf

    dmax = np.max(distances)
    
    if dmax < 1e-9:
        return np.inf

    ratio = dmin / dmax
    return -ratio

# Objective function for local refinement (Nelder-Mead) with a penalty for out-of-bounds points.
# This wrapper is crucial for unconstrained optimizers and is adopted from IP1.
def _objective_with_penalty(flat_points_array: np.ndarray, n: int, d: int) -> float:
    """Wrapper for _evaluate_points with a quadratic penalty for points outside [0,1] bounds."""
    # Quadratic penalty for being outside the [0, 1] hypercube
    penalty = np.sum(np.maximum(0, flat_points_array - 1)**2) + np.sum(np.maximum(0, -flat_points_array)**2)
    # A small tolerance check to avoid penalizing for minor floating point inaccuracies
    if penalty > 1e-12:
        # A large penalty multiplier guides the optimizer back into the valid region
        return 1e6 * penalty
    # If within bounds, return the actual objective function value
    return _evaluate_points(flat_points_array, n, d)

# Advanced initial population generator, inspired by the successful IP1.
# This incorporates the problem-specific "concentric circles" heuristic for N=16.
def _generate_initial_population(n_points: int, d_dims: int, popsize: int, seed: int) -> np.ndarray:
    """
    Generates a diverse initial population for differential_evolution by combining:
    1. Perturbed Grid: A very strong heuristic for this problem.
    2. Perturbed Concentric Circles (8+8 points for N=16): A key heuristic from IP1.
    3. Sobol Sequence: A quasi-random sequence for uniform coverage.
    4. Random Uniform: For additional diversity.
    """
    total_dims = n_points * d_dims
    initial_pop_array = np.zeros((popsize, total_dims))
    rng = np.random.default_rng(seed)

    # Allocate population slots based on IP1's successful distribution
    n_grid = int(popsize * 0.4)
    n_concentric_circles = int(popsize * 0.3)
    n_sobol = int(popsize * 0.2)
    
    current_idx = 0

    # 1. Perturbed Grid Strategy (for 4x4 case) - Centered grid points
    sqrt_n = int(np.sqrt(n_points))
    if sqrt_n * sqrt_n == n_points and d_dims == 2 and n_grid > 0:
        x_coords = np.linspace(0, 1, sqrt_n, endpoint=False) + 0.5 / sqrt_n
        y_coords = np.linspace(0, 1, sqrt_n, endpoint=False) + 0.5 / sqrt_n
        base_grid_points = np.array([(x, y) for y in y_coords for x in x_coords])
        jitter_scale = 0.05 / sqrt_n
        for i in range(n_grid):
            jitter = rng.uniform(-jitter_scale, jitter_scale, size=(n_points, d_dims))
            perturbed_points = np.clip(base_grid_points + jitter, 0, 1)
            initial_pop_array[current_idx] = perturbed_points.flatten()
            current_idx += 1

    # 2. Perturbed Concentric Circles Strategy (8+8 points for N=16) - from IP1
    if n_points == 16 and n_concentric_circles > 0:
        center = np.array([0.5, 0.5])
        r_inner, r_outer = 0.30, 0.45
        
        theta_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        inner_circle_base = center + r_inner * np.array([np.cos(theta_inner), np.sin(theta_inner)]).T
        
        theta_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8 # Staggered
        outer_circle_base = center + r_outer * np.array([np.cos(theta_outer), np.sin(theta_outer)]).T
        
        base_circles = np.vstack((inner_circle_base, outer_circle_base))
        jitter_magnitude = 0.03
        for i in range(n_concentric_circles):
            jitter = rng.uniform(-jitter_magnitude, jitter_magnitude, size=(n_points, d_dims))
            perturbed_points = np.clip(base_circles + jitter, 0, 1)
            initial_pop_array[current_idx] = perturbed_points.flatten()
            current_idx += 1

    # 3. Sobol Sequence Strategy
    if n_sobol > 0:
        sobol_sampler = qmc.Sobol(d=total_dims, seed=seed)
        sobol_sampler.fast_forward(100) # Advance to avoid initial low-quality points
        sobol_points = sobol_sampler.random(n=n_sobol)
        initial_pop_array[current_idx : current_idx + n_sobol] = sobol_points
        current_idx += n_sobol

    # 4. Random Uniform Strategy (fill remaining)
    n_random = popsize - current_idx
    if n_random > 0:
        initial_pop_array[current_idx:] = rng.uniform(0, 1, size=(n_random, total_dims))
        
    rng.shuffle(initial_pop_array)

    return initial_pop_array

def min_max_dist_dim2_16() -> np.ndarray:
    """ 
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    This implementation adopts the winning strategy from Inspiration Program 1, which successfully
    beat the benchmark. It uses:
    - A powerful global optimizer (Differential Evolution) with aggressive parameters.
    - A sophisticated, heuristic-based initial population including concentric circles.
    - A meticulous local refinement stage using Nelder-Mead with a penalty function.
    """
    n = 16
    d = 2
    seed = 42
    bounds = [(0, 1)] * (n * d)
    
    # Define population size, consistent with the successful IP1.
    de_popsize = 300
    
    # Generate the high-quality initial population using the concentric circles heuristic.
    initial_population = _generate_initial_population(n, d, de_popsize, seed=seed)

    # --- Phase 1: Global Search with Differential Evolution ---
    # Adopting the aggressive, parallelized settings from IP1.
    de_result = differential_evolution(
        func=_evaluate_points,
        bounds=bounds,
        args=(n, d),
        strategy='randtobest1bin',
        maxiter=25000,      # Increased iterations for a more exhaustive global search.
        popsize=de_popsize,
        tol=5e-8,           # Tighter tolerance to push DE to a higher precision result.
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False,       # Disable DE's internal polish to use our own local refiner.
        init=initial_population,
        seed=seed,
        workers=-1          # Use all available CPU cores.
    )

    # --- Phase 2: Local Refinement with Nelder-Mead ---
    # Using the robust, gradient-free Nelder-Mead with a penalty function, as in IP1.
    # This proved more effective than COBYLA for the final precision push.
    local_result = minimize(
        fun=_objective_with_penalty, # Use the objective with penalty for bounds handling.
        x0=de_result.x,
        args=(n, d),
        method='Nelder-Mead',
        options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
    )

    optimal_flat_points = local_result.x
    
    # Final clip to ensure points are strictly within [0,1] as a safeguard.
    optimal_points = np.clip(optimal_flat_points.reshape((n, d)), 0, 1)

    return optimal_points
# EVOLVE-BLOCK-END