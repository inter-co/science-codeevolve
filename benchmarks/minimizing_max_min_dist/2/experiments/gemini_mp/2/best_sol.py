# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution # Kept for potential future use or comparison, though not used in this version.
from scipy.spatial.distance import pdist
import cma # Added cma import for CMA-ES

# Objective function to be minimized
def _objective_function(points_flat):
    """
    Calculates the reciprocal of the min/max distance ratio for a set of points.
    This function is designed to be minimized by scipy.optimize and cma.

    Args:
        points_flat (np.ndarray): A 1D numpy array of 2*N coordinates (x1, y1, x2, y2, ...).

    Returns:
        float: 1 / (dmin / dmax), or np.inf if dmin is zero or invalid.
    """
    n_points = len(points_flat) // 2
    points = points_flat.reshape((n_points, 2))

    # Calculate all pairwise Euclidean distances
    distances = pdist(points)

    # Handle cases where there are fewer than 2 points or distances cannot be computed
    if len(distances) == 0:
        return np.inf

    dmin = np.min(distances)
    dmax = np.max(distances)

    # Penalize configurations where minimum distance is zero or extremely small
    # (i.e., overlapping or nearly overlapping points)
    # This also implicitly handles cases where dmax might be zero if all points are identical.
    if dmin <= 1e-12: # Using a small epsilon for numerical stability
        return np.inf # Return a very large value to heavily penalize invalid configurations

    ratio = dmin / dmax
    # The goal is to maximize the ratio, so we minimize its reciprocal
    return 1.0 / ratio

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions to maximize the min/max distance ratio.
    This is achieved using a multi-start Covariance Matrix Adaptation Evolution Strategy (CMA-ES),
    seeded with high-quality initial guesses based on a perturbed grid.

    Returns:
        points: np.ndarray of shape (16,2) containing the optimized coordinates.
    """

    n = 16 # Number of points
    d = 2  # Dimensions
    num_coords = n * d

    # CMA-ES parameters
    # sigma0: Initial standard deviation for the search distribution.
    # Increased slightly to allow for broader initial exploration around the perturbed grid.
    cma_sigma0 = 0.15 
    
    # Options for CMA-ES.
    # 'bounds': [min, max] for all dimensions, enforcing the unit square.
    # 'maxfevals': Maximum number of function evaluations. Adjusted to balance with increased restarts.
    #   Total evaluations: 8 restarts * 1.5M = 12M (vs previous 4 restarts * 2M = 8M).
    # 'popsize': Internal population size for CMA-ES.
    # 'verb_disp': Suppress verbose output from CMA-ES for cleaner execution.
    # 'seed': Will be set uniquely for each restart to ensure reproducibility and diversity.
    cma_options = {
        'bounds': [0, 1], 
        'maxfevals': 1500000, # Adjusted max function evaluations per restart
        'popsize': 100,      
        'verb_disp': -9,     
        'seed': None,        
    }

    # --- Multi-start CMA-ES with Smart Initialization ---
    # Increased number of restarts to improve the chance of finding the global optimum
    # by exploring more diverse regions of the search space.
    num_restarts = 8 

    best_objective_value = np.inf
    best_points = None

    for i in range(num_restarts):
        current_seed = 42 + i 
        rng = np.random.default_rng(current_seed)

        # Generate a high-quality initial guess (x0) for CMA-ES.
        # Start with a centered 4x4 grid, which is a strong heuristic for N=16.
        grid_size = int(np.sqrt(n))
        spacing = 1.0 / grid_size
        coords = np.linspace(spacing / 2.0, 1.0 - spacing / 2.0, grid_size)
        x, y = np.meshgrid(coords, coords)
        initial_grid_guess = np.vstack([x.ravel(), y.ravel()]).T.flatten()

        # Add a small, unique perturbation to x0 for each restart to promote diversity
        # and help CMA-ES explore different local basins.
        noise_scale = 0.05 * spacing 
        x0 = initial_grid_guess + rng.normal(loc=0, scale=noise_scale, size=num_coords)
        
        # Ensure the initial guess remains within the [0, 1] bounds.
        np.clip(x0, 0, 1, out=x0) 

        cma_options['seed'] = current_seed # Set seed for CMA-ES for reproducibility
        
        # Initialize and run CMA-ES
        es = cma.CMAEvolutionStrategy(x0, cma_sigma0, cma_options)
        es.optimize(_objective_function)
        
        # Get the best result from the CMA-ES run
        result_x = es.result.xbest
        result_fun = es.result.fbest

        if result_fun < best_objective_value:
            best_objective_value = result_fun
            best_points = result_x.reshape((n, d))
            
    if best_points is None:
        # Fallback in case no valid solution was found (highly unlikely)
        np.random.seed(42) 
        best_points = np.random.rand(n, d) 

    return best_points
# EVOLVE-BLOCK-END