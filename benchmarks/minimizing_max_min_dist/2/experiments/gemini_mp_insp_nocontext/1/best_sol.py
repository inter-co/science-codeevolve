# EVOLVE-BLOCK-START
import numpy as np 
# Removed scipy.spatial.distance.pdist as we'll use a Numba-jitted version
from scipy.optimize import differential_evolution, minimize
import numba # Added numba for JIT compilation

# Define constants for the number of points and dimensions.
N_POINTS = 16
DIMENSIONS = 2

@numba.jit(nopython=True, cache=True)
def _numba_pdist_sq(points: np.ndarray) -> np.ndarray:
    """
    Numba-jitted helper function to compute squared pairwise Euclidean distances efficiently.
    This replaces scipy.spatial.distance.pdist for Numba compatibility and performance.
    """
    n_points_local = points.shape[0]
    num_distances = n_points_local * (n_points_local - 1) // 2
    distances_sq = np.empty(num_distances, dtype=points.dtype)
    k = 0
    for i in range(n_points_local):
        for j in range(i + 1, n_points_local):
            diff = points[i] - points[j]
            distances_sq[k] = np.dot(diff, diff) # Squared Euclidean distance
            k += 1
    return distances_sq

def _objective(flat_coords: np.ndarray, n: int, d: int) -> float:
    """
    Objective function to minimize, adapted from the best-performing inspiration programs.
    It calculates -dmin/dmax using a fast, Numba-jitted squared distance calculation
    to avoid costly square root operations on the entire distance array.

    Args:
        flat_coords (np.ndarray): A 1D array of flattened point coordinates (x1, y1, x2, y2, ...).
        n (int): The number of points.
        d (int): The dimension of the space.

    Returns:
        float: The negative of the minimum to maximum distance ratio.
               Returns np.inf to penalize invalid configurations (e.g., coincident points).
    """
    points = flat_coords.reshape((n, d))
    
    # Use the fast Numba implementation to calculate squared distances.
    distances_sq = _numba_pdist_sq(points)

    if len(distances_sq) == 0:
        return np.inf

    dmin_sq = np.min(distances_sq)
    dmax_sq = np.max(distances_sq)

    # If dmax is effectively zero (all points identical), penalize heavily.
    if dmax_sq < 1e-12:
        return np.inf
    
    # If dmin is effectively zero (some points are coincident), this is a highly
    # undesirable configuration. Penalize it heavily to force the optimizer away.
    if dmin_sq < 1e-12:
        return np.inf

    # The ratio dmin/dmax is equivalent to sqrt(dmin_sq / dmax_sq).
    # This avoids taking the square root of the entire distance array, saving computation.
    ratio = np.sqrt(dmin_sq / dmax_sq)
    
    # The optimizer minimizes, so return the negative of the ratio to maximize it.
    return -ratio

def _jittered_grid_initialization(n_points: int, dimensions: int, pop_size_multiplier: int, bounds: list, seed_value: int) -> np.ndarray:
    """
    Generates a diverse initial population for differential_evolution using a jittered grid strategy,
    augmented with random points and the unperturbed grid for broader diversity, inspired by
    the best-performing inspiration program.
    
    Args:
        n_points (int): Number of points (e.g., 16).
        dimensions (int): Dimensionality of the space (e.g., 2).
        pop_size_multiplier (int): A multiplier to determine the base population size.
        bounds (list): List of (min, max) tuples for each variable.
        seed_value (int): Random seed for reproducibility.

    Returns:
        np.ndarray: An array representing the diverse initial population.
    """
    rng = np.random.default_rng(seed_value) # Use robust random number generator
    
    num_variables = n_points * dimensions
    population_size_base = pop_size_multiplier * num_variables # e.g., 20 * 16 * 2 = 640
    
    initial_population = []
    
    grid_side = int(np.sqrt(n_points))
    if grid_side * grid_side != n_points:
        raise ValueError("Jittered grid initialization currently supports only square grid numbers of points (e.g., 4x4 for 16 points).")

    min_coord, max_coord = bounds[0] # Assuming all variables have the same (0,1) bounds
    grid_cell_size = (max_coord - min_coord) / grid_side
    
    base_coords_1d = np.linspace(min_coord + grid_cell_size / 2, max_coord - grid_cell_size / 2, grid_side)
    base_grid_coords = np.array([[x, y] for y in base_coords_1d for x in base_coords_1d])

    # Generate jittered configurations for the bulk of the initial population
    for _ in range(population_size_base):
        jitter_amount = grid_cell_size * 0.45 # Optimal jitter from inspirations
        jitter = rng.uniform(-jitter_amount, jitter_amount, size=(n_points, dimensions))
        jittered_points = np.clip(base_grid_coords + jitter, min_coord, max_coord)
        initial_population.append(jittered_points.flatten())
    
    # Augment initial population with purely random configurations for diversity.
    num_random_configs = int(population_size_base * 0.1) # 10% random
    for _ in range(num_random_configs):
        initial_population.append(rng.uniform(min_coord, max_coord, num_variables))
    
    # Add the unperturbed grid configuration to ensure this stable starting point is considered.
    initial_population.append(base_grid_coords.flatten())

    return np.array(initial_population)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    This function implements a state-of-the-art, two-stage optimization strategy inspired by
    the best-performing programs.
    1. A thorough global search using Differential Evolution with a highly diverse initial
       population and a Numba-accelerated objective function.
    2. A high-precision local refinement of the best global solution using L-BFGS-B.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = N_POINTS
    d = DIMENSIONS
    bounds = [(0, 1)] * (n * d)
    seed_value = 42
    pop_size_multiplier = 20

    # --- Stage 1: Thorough Global Search with Differential Evolution ---
    initial_population = _jittered_grid_initialization(n, d, pop_size_multiplier, bounds, seed_value)

    # Parameters are tuned based on the best-performing inspiration program (Insp 1).
    de_result = differential_evolution(
        func=_objective,
        bounds=bounds,
        args=(n, d),
        strategy='randtobest1bin',
        maxiter=18000,          # A high number of iterations for a deep global search.
        recombination=0.9,
        tol=1e-8,               # Stricter tolerances for convergence.
        atol=1e-8,
        disp=False,
        polish=False,           # Disable DE's polish; we use a dedicated local search.
        workers=-1,
        seed=seed_value,
        init=initial_population
    )

    # --- Stage 2: Dedicated Local Refinement ---
    # Fine-tune the best solution from DE using a bound-constrained local optimizer.
    local_result = minimize(
        fun=_objective,
        x0=de_result.x,
        args=(n, d),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7} # Tighter tolerances for high precision
    )

    # Compare results and select the best one.
    if local_result.fun < de_result.fun:
        optimized_params = local_result.x
    else:
        optimized_params = de_result.x

    # Reshape the optimized flattened coordinates back into an (n, d) array of points.
    optimized_points = optimized_params.reshape((n, d))

    return optimized_points
# EVOLVE-BLOCK-END