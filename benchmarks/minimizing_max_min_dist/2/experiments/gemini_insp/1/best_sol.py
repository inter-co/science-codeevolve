# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize # Added minimize for local refinement
from scipy.stats import qmc

# Helper function to evaluate the min/max distance ratio, adapted from inspirations.
def _evaluate_points(flat_points_array: np.ndarray, n: int, d: int) -> float:
    """
    Evaluates the min/max distance ratio for a given set of points.
    The optimizer minimizes, so we return the negative ratio.
    """
    points = flat_points_array.reshape((n, d))
    distances = pdist(points)

    # Use np.inf for severe penalties, as is standard practice for scipy.optimize
    if len(distances) == 0:
        return np.inf # Severe penalty for invalid point configurations (e.g., n=1)

    dmin = np.min(distances)
    
    # Penalize configurations with overlapping points heavily.
    if dmin < 1e-9:
        return np.inf # Severe penalty

    dmax = np.max(distances)
    # Penalize if dmax is extremely small, meaning points are clustered in a tiny area,
    # which implies poor dispersion. This is a robust fallback for tiny clusters.
    if dmax < 1e-9:
        return np.inf # Severe penalty

    ratio = dmin / dmax
    return -ratio # Minimize -ratio to maximize ratio

# Advanced initial population generator, inspired by Inspiration Program 1.
# Objective function for local refinement (Nelder-Mead) with a penalty for out-of-bounds points.
# Adapted from Inspiration 3 for explicit local refinement.
def _objective_with_penalty(flat_points_array: np.ndarray, n: int, d: int) -> float:
    """Wrapper for _evaluate_points with a quadratic penalty for points outside [0,1] bounds."""
    penalty = np.sum(np.maximum(0, flat_points_array - 1)**2) + np.sum(np.maximum(0, -flat_points_array)**2)
    if penalty > 1e-12: # If any point is significantly out of bounds, apply a large penalty
        return 1e6 * penalty # Return a large positive value to guide optimizer back to bounds.
    return _evaluate_points(flat_points_array, n, d)

# Advanced initial population generator, inspired by Inspiration Program 1.
def _generate_initial_population(n_points: int, d_dims: int, popsize: int, seed: int) -> np.ndarray:
    """
    Generates a diverse initial population for differential_evolution by combining:
    1. Perturbed Grid: A very strong heuristic for this problem. (Improved centering from Insp 1)
    2. Perturbed Concentric Circles (8+8 points for N=16): Explores boundary-heavy solutions
       with a specific symmetry. (Adapted from Inspiration 1).
    3. Sobol Sequence: A quasi-random sequence for uniform coverage.
    4. Random Uniform: For additional diversity.
    """
    total_dims = n_points * d_dims
    initial_pop_array = np.zeros((popsize, total_dims))
    rng = np.random.default_rng(seed)

    # Allocate population slots to different strategies (Adapted proportions from Insp 1)
    n_grid = int(popsize * 0.4) # Increased grid proportion
    n_concentric_circles = int(popsize * 0.3) # Specific for N=16
    n_sobol = int(popsize * 0.2)
    n_random = popsize - n_grid - n_concentric_circles - n_sobol
    
    # Ensure non-negative counts and sum to popsize (Robustness from Insp 1)
    n_grid = max(0, n_grid)
    n_concentric_circles = max(0, n_concentric_circles)
    n_sobol = max(0, n_sobol)
    n_random = popsize - (n_grid + n_concentric_circles + n_sobol)
    n_random = max(0, n_random) # Final adjustment

    current_idx = 0

    # 1. Perturbed Grid Strategy (for 4x4 case) - Centered grid points (from Insp 1)
    sqrt_n = int(np.sqrt(n_points))
    if sqrt_n * sqrt_n == n_points and d_dims == 2 and n_grid > 0: # Added n_grid > 0 check for robustness (from Insp 3)
        x_coords = np.linspace(0, 1, sqrt_n, endpoint=False) + 0.5 / sqrt_n
        y_coords = np.linspace(0, 1, sqrt_n, endpoint=False) + 0.5 / sqrt_n
        base_grid_points = np.array([(x, y) for y in y_coords for x in x_coords])
        jitter_scale = 0.05 / sqrt_n # Smaller jitter for a good base (from Insp 1)
        for i in range(n_grid):
            if current_idx >= popsize: break # Ensure not to exceed popsize
            jitter = rng.uniform(-jitter_scale, jitter_scale, size=(n_points, d_dims))
            perturbed_points = np.clip(base_grid_points + jitter, 0, 1)
            initial_pop_array[current_idx] = perturbed_points.flatten()
            current_idx += 1

    # 2. Perturbed Concentric Circles Strategy (8+8 points for N=16) - Adapted from Inspiration 1
    if n_points == 16 and n_concentric_circles > 0: # Added n_points == 16 and n_concentric_circles > 0 check for robustness (from Insp 3)
        center = np.array([0.5, 0.5])
        r_inner, r_outer = 0.30, 0.45 # Tuned radii based on common optimal configurations (from Insp 1)
        
        theta_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        inner_circle_base = center + r_inner * np.array([np.cos(theta_inner), np.sin(theta_inner)]).T
        
        theta_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8 # Stagger for better dispersion (from Insp 1)
        outer_circle_base = center + r_outer * np.array([np.cos(theta_outer), np.sin(theta_outer)]).T
        
        base_circles = np.vstack((inner_circle_base, outer_circle_base))
        jitter_magnitude = 0.03 # Slightly reduced jitter for this strong heuristic (from Insp 1)
        for i in range(n_concentric_circles):
            if current_idx >= popsize: break # Ensure not to exceed popsize
            jitter = rng.uniform(-jitter_magnitude, jitter_magnitude, size=(n_points, d_dims))
            perturbed_points = np.clip(base_circles + jitter, 0, 1)
            initial_pop_array[current_idx] = perturbed_points.flatten()
            current_idx += 1

    # 3. Sobol Sequence Strategy
    if n_sobol > 0: # Added n_sobol > 0 check for robustness (from Insp 3)
        sobol_sampler = qmc.Sobol(d=total_dims, seed=seed)
        sobol_sampler.fast_forward(100) # Advance to avoid initial degenerate points (from Insp 1)
        sobol_points = sobol_sampler.random(n=n_sobol)
        initial_pop_array[current_idx : current_idx + n_sobol] = sobol_points
        current_idx += n_sobol

    # 4. Random Uniform Strategy
    if n_random > 0: # Added n_random > 0 check for robustness (from Insp 3)
        initial_pop_array[current_idx:] = rng.uniform(0, 1, size=(n_random, total_dims))
        
    # Shuffle the combined population to ensure diversity is evenly distributed (from Insp 1)
    rng.shuffle(initial_pop_array)

    return initial_pop_array

def min_max_dist_dim2_16() -> np.ndarray:
    """ 
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    This implementation combines the best strategies from analysis:
    - A powerful global optimizer (Differential Evolution).
    - A sophisticated, heuristic-based initial population to guide the search.
    - Aggressive optimization parameters (large population, many iterations).
    - Final local refinement using `polish=True`.
    """
    n = 16
    d = 2
    seed = 42
    bounds = [(0, 1)] * (n * d)
    
    # Define a consistent population size for both initial generation and DE.
    de_popsize = 300 # Increased population size for greater diversity and robustness in the final push.
    
    # Generate a diverse initial population using multiple geometric and quasi-random heuristics.
    initial_population = _generate_initial_population(n, d, de_popsize, seed=seed)

    # --- Phase 1: Global Search with Differential Evolution ---
    # Perform differential evolution with aggressive, parallelized settings.
    # Set polish=False to avoid DE's default gradient-based local refinement,
    # and instead use a dedicated gradient-free local optimizer.
    de_result = differential_evolution(
        func=_evaluate_points,
        bounds=bounds,
        args=(n, d),
        strategy='randtobest1bin', # Balances exploration and exploitation, proven effective.
        maxiter=25000,      # Significantly increased iterations for an even more exhaustive global search.
        popsize=de_popsize, # Consistent and increased population size.
        tol=5e-8,           # Tighter relative tolerance for maximum precision convergence (from Insp 2 & 3).
        mutation=(0.5, 1.5), # Effective mutation range for exploration and fine-tuning.
        recombination=0.9,  # High recombination rate for efficient mixing of promising solutions.
        polish=False,       # Disable DE's internal local optimizer.
        init=initial_population,
        seed=seed,
        workers=-1          # Use all available CPU cores.
    )

    # --- Phase 2: Local Refinement with Nelder-Mead ---
    # Use the best result from DE as the starting point for a meticulous local search.
    # Nelder-Mead is gradient-free and robust to non-smooth objectives, and _objective_with_penalty
    # provides necessary bounds handling for this unconstrained optimizer.
    local_result = minimize(
        fun=_objective_with_penalty, # Use objective with penalty for Nelder-Mead bounds handling.
        x0=de_result.x,
        args=(n, d),
        method='Nelder-Mead',
        options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True} # Aggressive options for precision.
    )

    optimal_flat_points = local_result.x
    
    # Final clip to ensure points are strictly within [0,1] due to potential float inaccuracies
    # and to enforce the bounds strictly after optimization.
    optimal_points = np.clip(optimal_flat_points.reshape((n, d)), 0, 1)

    return optimal_points
# EVOLVE-BLOCK-END