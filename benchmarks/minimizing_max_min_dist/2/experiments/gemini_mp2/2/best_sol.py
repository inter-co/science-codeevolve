# EVOLVE-BLOCK-START
import numpy as np 
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize

def calculate_min_max_ratio(points: np.ndarray) -> float:
    """
    Calculates the ratio of minimum to maximum distance among a set of points.
    
    Args:
        points: np.ndarray of shape (N, D) containing the coordinates of the points.
        
    Returns:
        float: The min_max_ratio. Returns 0 if dmax is zero or dmin is zero.
    """
    if points.shape[0] < 2:
        return 0.0 # Not enough points to calculate distances

    # Compute all pairwise Euclidean distances
    distances = pdist(points, 'euclidean')

    if len(distances) == 0: # Handle case where pdist returns empty (e.g., N=1)
        return 0.0

    dmin = np.min(distances)
    dmax = np.max(distances)

    if dmax == 0:
        # All points are identical, which is a very bad configuration.
        # Returning 0.0 here, which will be converted to np.inf by the objective function.
        return 0.0 
    
    return dmin / dmax

def objective(points_flat: np.ndarray) -> float:
    """
    Objective function for optimization, returns the negative of the min_max_ratio.
    
    Args:
        points_flat: 1D NumPy array of flattened point coordinates [x1, y1, x2, y2, ...].
        
    Returns:
        float: Negative of the min_max_ratio. Returns np.inf if the ratio is 0.0.
    """
    n = 16
    d = 2
    points = points_flat.reshape((n, d))
    
    ratio = calculate_min_max_ratio(points)
    
    # We want to maximize the ratio, so we minimize its negative.
    # If ratio is 0.0 (e.g., all points identical, or <2 points), penalize heavily.
    if ratio == 0.0:
        return np.inf # Penalize invalid configurations with a very large positive number
    
    return -ratio

def objective_log(points_flat: np.ndarray) -> float:
    """
    Objective function based on minimizing log(dmax) - log(dmin), which is
    equivalent to maximizing dmin / dmax. This logarithmic form can create a
    smoother landscape for optimizers.
    """
    n = 16
    d = 2
    points = points_flat.reshape((n, d))
    
    if points.shape[0] < 2:
        return np.inf

    distances = pdist(points, 'euclidean')

    if len(distances) == 0:
        return np.inf

    # Add a small epsilon for numerical stability
    epsilon = 1e-9
    dmin = np.min(distances)
    dmax = np.max(distances)

    if dmin < epsilon: # Penalize configurations with overlapping or very close points
        return np.inf

    return np.log(dmax + epsilon) - np.log(dmin + epsilon)


def energy_objective(points_flat: np.ndarray, n: int, d: int, w_dmax: float) -> float:
    """
    Objective function combining repulsive energy with a reward for large dmax.
    Minimizing this value pushes points apart (energy term) and encourages
    them to spread out to fill the space (dmax term).
    """
    points = points_flat.reshape((n, d))
    distances = pdist(points, 'euclidean')
    
    if distances.size == 0:
        return np.inf

    dmax = np.max(distances)
    # Use 1/d^2 for stronger repulsion. Add epsilon for stability.
    energy = np.sum(1.0 / (distances**2 + 1e-9))
    
    # We minimize this function. A larger dmax results in a smaller (better)
    # objective value, thus rewarding configurations that are spread out.
    return energy - w_dmax * dmax

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions to maximize the min/max distance ratio using a multi-stage approach.
    1. Pre-optimization: Uses an energy model to find a good initial distribution.
    2. Targeted Seeding: Creates a population for DE based on the pre-optimized result.
    3. Global Optimization: Runs Differential Evolution on the true min/max ratio objective.
    4. Local Refinement: Fine-tunes the best global solution.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    bounds = [(0, 1)] * (n * d)
    seed = 42
    rng = np.random.default_rng(seed)

    # --- Stage 1: Pre-optimization using a Repulsive Energy Model ---
    # Start with a structured guess (a 4x4 grid) and optimize it using the energy model.
    # This quickly finds a high-quality, well-distributed initial configuration.
    grid_size = int(np.sqrt(n))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    initial_grid = np.vstack((xv.ravel(), yv.ravel())).T.flatten()
    
    # For pre-optimization, we focus on pure repulsion to get a good, uniform initial spread.
    # Setting w_dmax to 0.0 effectively removes the dmax reward from the energy function.
    w_dmax = 0.0 

    pre_opt_result = minimize(
        energy_objective,
        x0=initial_grid,
        args=(n, d, w_dmax),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 10000} # Further increased iterations for better pre-optimization
    )
    pre_optimized_config = pre_opt_result.x

    # --- Stage 2: Create a Targeted Initial Population for Differential Evolution ---
    # Seed the population with our high-quality guess and its variations to guide the search.
    # Significantly increased population size for DE to enhance global exploration.
    de_popsize = 12 * n * d  # Heuristic: 12 * dimensions (32) = 384
    initial_pop = np.zeros((de_popsize, n * d))
    initial_pop[0] = pre_optimized_config  # The best guess is the first member
    
    # Create perturbed versions of the best guess for local exploration
    num_perturbed = de_popsize // 2 - 1
    for i in range(1, num_perturbed + 1):
        # Perturbation range to encourage broader exploration around the seed
        perturbation = rng.uniform(-0.07, 0.07, n * d) 
        initial_pop[i] = np.clip(pre_optimized_config + perturbation, 0, 1)

    # Fill the rest with random configurations for global diversity
    for i in range(num_perturbed + 1, de_popsize):
        initial_pop[i] = rng.random(n * d)

    # --- Stage 3: Global Optimization targeting the log-ratio objective ---
    # The log-ratio objective can provide a smoother landscape for the optimizer.
    # Adjusted maxiter and popsize for better exploration/exploitation balance.
    global_result = differential_evolution(
        objective_log, 
        bounds, 
        init=initial_pop, # Use the custom seeded population
        seed=seed,
        maxiter=15000, # Adjusted maxiter to balance with increased popsize
        strategy='randtobest1bin',
        mutation=0.7,
        recombination=0.85,
        tol=1e-6, # Stricter tolerance for convergence
        disp=False,
        workers=-1,
        polish=True # Enable internal polish for DE's best solution
    )
    
    # --- Stage 4: Final Local Refinement ---
    # Use the best solution from DE and fine-tune it with a precise local optimizer on the same log objective.
    local_result = minimize(
        objective_log, 
        x0=global_result.x, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 30000} 
    )
    
    optimized_points_flat = local_result.x
    optimized_points = optimized_points_flat.reshape((n, d))

    return optimized_points
# EVOLVE-BLOCK-END