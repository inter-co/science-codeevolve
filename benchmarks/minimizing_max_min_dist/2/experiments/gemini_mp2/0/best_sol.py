# EVOLVE-BLOCK-START
import numpy as np 
# Removed pdist as we'll use a Numba-jitted custom distance function for performance
from scipy.optimize import dual_annealing
from numba import jit # Added numba for JIT compilation

@jit(nopython=True)
def _calculate_min_max_distances_numba(points: np.ndarray) -> (float, float):
    """
    Numba-jitted helper function to efficiently calculate minimum and maximum
    Euclidean distances between all unique pairs of points.
    This avoids the overhead of creating a full distance matrix and then finding min/max.
    """
    n_points = points.shape[0]
    dmin_sq = np.inf
    dmax_sq = 0.0
    
    # For N=16, we always have at least 2 points, so this check is primarily for robustness
    if n_points < 2:
        return 0.0, 0.0 # Should not occur for N=16, but handles edge case.

    for i in range(n_points):
        for j in range(i + 1, n_points): # Iterate over unique pairs
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < dmin_sq:
                dmin_sq = dist_sq
            if dist_sq > dmax_sq:
                dmax_sq = dist_sq
    
    # If dmax_sq is 0, it implies all points are at the exact same location.
    # We return 0.0 for both dmin and dmax, which will trigger a penalty in _objective.
    if dmax_sq == 0.0:
        return 0.0, 0.0 
    
    return np.sqrt(dmin_sq), np.sqrt(dmax_sq)


def _objective(flat_points: np.ndarray) -> float:
    """
    Objective function to minimize for dual_annealing.
    It calculates -(dmin/dmax) for a flattened array of points.
    """
    n_points = 16
    dimensions = 2
    points = flat_points.reshape((n_points, dimensions))
    
    # Calculate min and max distances using the Numba-jitted helper
    dmin, dmax = _calculate_min_max_distances_numba(points)
    
    # Penalize configurations where dmax is zero (all points are identical).
    # This also covers the case where _calculate_min_max_distances_numba returns 0.0, 0.0.
    if dmax == 0.0:
        return np.inf # Heavily penalize for all points coincident.
    
    # Penalize configurations where dmin is zero (at least two points are coincident).
    # A dmin/dmax ratio of 0 is a very poor solution for point dispersion.
    if dmin == 0.0:
        return np.inf # Heavily penalize for overlapping points.
    
    # We want to maximize dmin/dmax, so we minimize its negative.
    return -(dmin / dmax)

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    
    """

    n = 16
    d = 2

    # Set random seed for reproducibility for initial point generation
    np.random.seed(42)
    
    # Initial guess: A 4x4 grid, which is a strong starting point for N=16.
    # This provides a much better initial configuration than random points,
    # guiding the optimizer towards a high-quality solution more quickly.
    grid_coords = np.linspace(0.0, 1.0, int(np.sqrt(n)))
    x, y = np.meshgrid(grid_coords, grid_coords)
    x0 = np.vstack([x.ravel(), y.ravel()]).T.flatten() 
    
    # Define bounds for each coordinate to keep points within [0,1]x[0,1].
    # There are n*d coordinates, each bounded between 0 and 1.
    bounds = [(0.0, 1.0)] * (n * d)
    
    # Apply dual_annealing for global optimization.
    # The 'seed' parameter for dual_annealing also ensures reproducibility of its internal stochasticity.
    # maxiter is set to 1000 for a balance between solution quality and evaluation time.
    # Apply dual_annealing for global optimization.
    # Increased maxiter significantly to allow for more thorough exploration, aiming for a better global optimum.
    # Increased initial_temp to encourage broader initial exploration and better escape from local minima.
    # The 'seed' parameter for dual_annealing also ensures reproducibility of its internal stochasticity.
    # Apply dual_annealing for global optimization.
    # The 'seed' parameter for dual_annealing ensures reproducibility of its internal stochasticity.
    # Increased maxiter to allow for deeper search and convergence.
    # Reduced initial_temp significantly. A high initial_temp was degrading the good initial grid (x0).
    # A lower initial_temp promotes refinement around the strong starting point rather than aggressive, chaotic exploration.
    # Apply dual_annealing for global optimization.
    # The 'seed' parameter for dual_annealing ensures reproducibility of its internal stochasticity.
    # Maxiter is kept high to allow for thorough refinement and convergence.
    # Initial_temp is drastically reduced. Starting with a good initial guess (4x4 grid)
    # means we want the optimizer to refine it, not aggressively explore or "melt" it.
    # A low initial_temp focuses the annealing process on local improvements around x0.
    # Define minimizer_kwargs for local search within dual_annealing.
    # Using SLSQP, which is often more robust for bounded optimization problems than the default L-BFGS-B.
    # Increased local 'maxiter' to allow for more thorough refinement during each local search step.
    minimizer_kwargs = {
        "method": "SLSQP", 
        "options": {"maxiter": 1000, "ftol": 1e-10} # Increased local maxiter and added function tolerance for higher precision.
    }

    # Apply dual_annealing for global optimization.
    # The 'seed' parameter for dual_annealing ensures reproducibility of its internal stochasticity.
    # Maxiter further increased to leverage Numba speedup and provide more global search iterations.
    # Initial_temp remains at the default (10.0 for [0,1] bounds) for a balanced annealing schedule.
    result = dual_annealing(
        func=_objective, 
        bounds=bounds, 
        x0=x0, 
        seed=42,
        maxiter=60000, # Further increased global maxiter to maximize search depth within time limit.
        initial_temp=10.0, # Retain default for balanced exploration/refinement.
        minimizer_kwargs=minimizer_kwargs # Pass custom local minimizer settings for better refinement.
    )
    
    # Reshape the optimized 1D array of coordinates back into an (n, d) array of points.
    optimized_points = result.x.reshape((n, d))

    return optimized_points
# EVOLVE-BLOCK-END