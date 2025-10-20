# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing # Replaced differential_evolution, minimize with dual_annealing

def _objective_function(coords_flat):
    """
    Objective function for optimization.
    Takes a flattened array of 2D point coordinates and returns the negative
    of the min_max_ratio, as scipy.optimize functions minimize.
    """
    n = 16
    d = 2
    points = coords_flat.reshape((n, d))

    distances = pdist(points)

    # Handle cases where distances might be empty or all zero (e.g., initial bad guesses)
    if len(distances) == 0:
        return 1.0 # Return a large value to penalize empty or degenerate distance sets

    dmin = np.min(distances)
    dmax = np.max(distances)

    # CRITICAL FIX: Penalize zero or near-zero dmin
    # If dmin is very small (points are effectively overlapping),
    # the ratio dmin/dmax approaches 0. Minimizing -dmin/dmax would push
    # the optimizer towards dmin=0, which is undesired as it results in a 0 ratio.
    # We return a large positive value to represent a very bad outcome.
    if dmin <= 1e-9: # Threshold for considering points overlapping
        return 1.0 # A large positive value to penalize dmin=0, much worse than 0 or negative values

    if dmax == 0: # All points are identical, which is a very bad solution
        return 1.0

    return -dmin / dmax

def min_max_dist_dim2_16()->np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses dual_annealing for global and local search, starting with a 4x4 grid initialization.
    Points are constrained to the unit square [0,1]x[0,1].

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2

    # Define bounds for each coordinate (x, y) for all 16 points within the unit square [0,1]x[0,1]
    bounds = [(0.0, 1.0)] * (n * d)

    # 1. Initialization: Start with a 4x4 grid and add jitter
    # This provides a strong initial guess for 16 points in a square,
    # significantly improving the starting point for global optimization.
    grid_points = np.array([(i/3, j/3) for i in range(4) for j in range(4)]) # (16, 2)
    
    # Add random jitter to break perfect symmetry and help escape local minima
    # Using a dedicated RNG for reproducibility of jitter
    rng = np.random.default_rng(seed=42) 
    jitter_amount = 0.02 # Small perturbation, e.g., +/- 0.02
    jitter = (rng.random(n * d) - 0.5) * 2 * jitter_amount # uniform in [-jitter_amount, jitter_amount]
    
    # Apply jitter and ensure points stay within [0,1] bounds
    initial_guess_flat = (grid_points.flatten() + jitter).clip(0, 1) 

    # 2. Global and Local Search with Dual Annealing
    # Dual Annealing is a robust global optimization algorithm suitable for non-smooth functions
    # and functions with many local minima. It combines generalized simulated annealing
    # with a local search method.
    # We use Nelder-Mead for local search because it is gradient-free and robust
    # to non-smoothness, which is present due to np.min/np.max in the objective function.

    # Set a fixed seed for reproducibility.
    # 'maxiter', 'initial_temp', 'visit', 'accept' are tuned for a balance
    # between solution quality and computation time.
    da_result = dual_annealing(
        _objective_function,
        bounds,
        seed=42, # Reproducibility
        x0=initial_guess_flat, # Start from the 4x4 grid configuration with jitter
        maxiter=5000, # Increased iterations for more thorough global search (from 2500)
        initial_temp=5230.0, # Kept default initial temperature for broad exploration
        restart_temp_ratio=2e-5, # Kept default restart temperature ratio
        visit=2.62, # Kept default visit parameter
        accept=-5.0, # Kept default accept parameter
        minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'tol': 1e-6}}, # Use Nelder-Mead for local search with more iterations (from 1000)
        # disp=True # Uncomment to display optimization progress
    )

    # Reshape the flattened optimized coordinates back into (16, 2)
    optimized_coords_flat = da_result.x
    optimized_points = optimized_coords_flat.reshape((n, d))

    return optimized_points
# EVOLVE-BLOCK-END