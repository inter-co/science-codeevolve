# EVOLVE-BLOCK-START
import numpy as np 
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize

def _objective(flat_points: np.ndarray) -> float:
    """
    Objective function to minimize for dual_annealing.
    It calculates -(dmin/dmax) for a flattened array of points.
    """
    n_points = 16
    dimensions = 2
    points = flat_points.reshape((n_points, dimensions))
    
    # Calculate all unique pairwise Euclidean distances
    distances = pdist(points)
    
    # If there are no distances (e.g., fewer than 2 points, though n_points is fixed at 16 here),
    # or if all points are coincident resulting in zero distances, return a high penalty.
    if len(distances) == 0:
        return np.inf
        
    dmin = np.min(distances)
    dmax = np.max(distances)
    
    # Penalize configurations where dmax is zero (all points are identical)
    if dmax == 0:
        return np.inf
    
    # We want to maximize dmin/dmax, so we minimize its negative
    return -(dmin / dmax)

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    
    """

    n = 16
    d = 2

    # Set random seed for reproducibility for all stochastic components
    np.random.seed(42)
    
    # Define bounds for each coordinate to keep points within [0,1]x[0,1].
    # There are n*d coordinates, each bounded between 0 and 1.
    bounds = [(0.0, 1.0)] * (n * d)

    # Define minimizer_kwargs for local search steps within dual_annealing
    # and for the final local refinement stage.
    # SLSQP is generally robust for bounded optimization.
    minimizer_kwargs_da = {
        "method": "SLSQP", 
        "options": {"maxiter": 500, "ftol": 1e-8} # Enhanced local search precision
    }

    # Stage 1: Global search using dual_annealing.
    # Start from an intelligent initial configuration (4x4 grid) as recommended
    # for N=16, rather than a purely random start. This significantly improves
    # the baseline and guides the global search more effectively.
    
    # Generate a 4x4 grid as initial guess
    grid_side = int(np.sqrt(n)) # For N=16, this is 4
    x_coords = np.linspace(0, 1, grid_side)
    y_coords = np.linspace(0, 1, grid_side)
    
    # Create grid points
    initial_points_grid = np.array([(x, y) for y in y_coords for x in x_coords])
    
    # Add a small random perturbation (jitter) to the initial grid points.
    # This helps break perfect symmetry and allows the optimizer to explore
    # slight deviations from the grid, which might be optimal.
    perturbation_scale = 0.01 # Small scale, e.g., 1% of unit length relative to unit square side
    jitter = np.random.uniform(-perturbation_scale, perturbation_scale, initial_points_grid.shape)
    
    # Apply jitter and ensure points stay within bounds [0,1]
    x0_global_search = (initial_points_grid + jitter).flatten()
    x0_global_search = np.clip(x0_global_search, 0.0, 1.0) # Ensure points remain within the unit square
    
    global_result = dual_annealing(
        func=_objective, 
        bounds=bounds, 
        x0=x0_global_search, # Start from intelligent, jittered 4x4 grid
        seed=42,
        maxiter=15000, # Max iterations for the global search stage.
        initial_temp=1e4, # High initial temperature for wide exploration.
        minimizer_kwargs=minimizer_kwargs_da # Use configured local minimizer for steps within annealing.
    )
    
    # Stage 2: Local refinement using scipy.optimize.minimize.
    # This stage takes the best solution found by dual_annealing and performs
    # a highly precise local optimization to fine-tune the point arrangement.
    optimized_points_flat_initial = global_result.x

    final_result = minimize(
        fun=_objective,
        x0=optimized_points_flat_initial,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 1500, 'ftol': 1e-10} # High maxiter and ftol for high precision local refinement.
    )
    
    # Reshape the optimized 1D array of coordinates back into an (n, d) array of points.
    optimized_points = final_result.x.reshape((n, d))

    return optimized_points
# EVOLVE-BLOCK-END