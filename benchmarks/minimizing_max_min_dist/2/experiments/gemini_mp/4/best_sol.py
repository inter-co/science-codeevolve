# EVOLVE-BLOCK-START
import numpy as np 
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

def objective(coords_flat: np.ndarray) -> float:
    """
    Objective function for differential evolution.
    Minimizes the negative ratio of minimum to maximum distance.
    
    Args:
        coords_flat: A 1D numpy array of shape (N*D,) representing flattened coordinates
                     (e.g., (x1, y1, x2, y2, ...)).
                     For N=16, D=2, this is (32,).

    Returns:
        The negative of the dmin/dmax ratio. `differential_evolution` minimizes this value.
    """
    n_points = 16
    coords = coords_flat.reshape(n_points, 2)

    # Calculate all pairwise Euclidean distances
    # pdist returns a condensed distance matrix (1D array of unique distances)
    distances = pdist(coords)

    # If there are no distances (should not happen with n_points=16), return a very bad score.
    if len(distances) == 0:
        return np.inf # Penalize configurations with no distances (e.g., less than 2 points)

    dmin = np.min(distances)
    dmax = np.max(distances)

    # CRITICAL FIX: Penalize configurations where points coincide or are extremely close.
    # As per problem statement: "If dmin is zero (meaning two or more points coincide or are extremely close, e.g., dmin < 1e-9),
    # return a very large number (e.g., np.inf or 1e10) to heavily penalize such configurations".
    # This prevents the optimizer from converging to solutions with dmin=0 (ratio=0), which was a bug,
    # and ensures the objective function correctly penalizes degenerate solutions.
    if dmin < 1e-9: # Use a small epsilon for robustness with floating point comparisons
        return np.inf 
    
    # If dmax is effectively zero, it means all points are at the exact same location.
    # This implies dmin is also zero (or near zero). This case is now correctly handled
    # by the `if dmin < 1e-9` check. The original code's `if dmax == 0: return -np.inf`
    # was incorrect (should be np.inf for minimization) and is now redundant.
    # Removing it for clarity and correctness.
    
    ratio = dmin / dmax
    return -ratio # differential_evolution minimizes, so we minimize -ratio to maximize ratio

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    This is achieved using scipy.optimize.differential_evolution to find the optimal point arrangement.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 optimal points.
    """
    from scipy.stats import qmc

    n = 16 # Number of points
    d = 2  # Dimensions

    # Define bounds for each coordinate. All x and y coordinates must be within [0, 1].
    bounds = [(0, 1)] * (n * d)
    n_coords = n * d

    # Determine population size.
    de_popsize_multiplier = 40 # Further increased population size for enhanced exploration
    actual_pop_size = de_popsize_multiplier * n_coords # 40 * 32 = 1280

    # --- Custom Initial Population Generation (Hybrid Approach) ---
    # Generate a base 4x4 grid of points. This provides a strong starting configuration.
    x_grid = np.linspace(0, 1, 4)
    y_grid = np.linspace(0, 1, 4)
    base_grid_points = np.array([(x, y) for y in y_grid for x in x_grid])
    base_grid_flat = base_grid_points.flatten()

    initial_population = np.zeros((actual_pop_size, n_coords))
    
    # Use a dedicated RandomState for reproducibility of the perturbations.
    rng = np.random.RandomState(42)

    # 1. Majority of population (80%): Quasi-random Sobol sequence
    # Provides superior, more uniform coverage of the search space compared to uniform random,
    # increasing the chance of finding completely different, potentially better optima.
    # This prioritizes global exploration over initial bias towards grid-like configurations.
    num_sobol_individuals = int(0.8 * actual_pop_size)
    sampler = qmc.Sobol(d=n_coords, seed=42)
    sobol_points = sampler.random(n=num_sobol_individuals)
    initial_population[:num_sobol_individuals] = sobol_points

    # 2. Minority of population (20%): Perturbed grid
    # This still provides seeds for configurations near a known good starting point.
    perturbation_scale = 0.05 
    num_grid_individuals = actual_pop_size - num_sobol_individuals
    for i in range(num_grid_individuals):
        perturbation = rng.uniform(-perturbation_scale, perturbation_scale, n_coords)
        perturbed_coords = base_grid_flat + perturbation
        initial_population[num_sobol_individuals + i] = np.clip(perturbed_coords, 0, 1)
    # --- End Custom Initial Population Generation ---

    # Use scipy.optimize.differential_evolution for global optimization.
    # - strategy: 'randtobest1bin' offers a good balance of exploration and exploitation.
    # - recombination: Explicitly set to 0.9 for faster convergence.
    # - maxiter: Significantly increased for more thorough exploration.
    # - tol: Tightened tolerance for more precise convergence.
    # - polish: Set to False as the objective function is non-smooth, and gradient-based
    #           local optimization might not be effective or could get stuck in local minima.
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=20000, # Slightly reduced maxiter to accommodate the 'polish' step within eval_time
        strategy='randtobest1bin', 
        recombination=0.9,
        tol=1e-5,    # Tightened tolerance for more precise convergence
        disp=False,    
        init=initial_population, 
        workers=-1,
        polish=True # Enabled polish for fine-tuning the best solution
    )

    # The optimal coordinates are stored in result.x (a 1D array).
    # Reshape them back into a (16, 2) array of points.
    optimal_points = result.x.reshape(n, d)

    return optimal_points
# EVOLVE-BLOCK-END