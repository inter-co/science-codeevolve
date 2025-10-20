# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2D to maximize the ratio of minimum to maximum distance
    using a global optimization algorithm (Dual Annealing) with a structured initial guess.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    seed = 42

    # Objective function to minimize: the negative of the ratio d_min / d_max.
    # Minimizing -ratio is equivalent to maximizing the ratio.
    def objective_func(x, n_points, dim):
        # Reshape the flat array of variables into a (n, d) array of points
        points = x.reshape(n_points, dim)
        
        # Calculate all pairwise distances efficiently
        dists = pdist(points)
        
        # If any points are coincident (d_min ~ 0), return a very large penalty value.
        # This strongly discourages solutions where points are too close. Using a large
        # finite number can be more stable for some optimizers than np.inf.
        if not np.all(dists > 1e-9):
            return 1e6

        d_min = np.min(dists)
        d_max = np.max(dists)

        # Avoid division by zero or extremely small d_max, which indicates collapsed points.
        if d_max < 1e-9:
            return 1e6

        return -d_min / d_max

    # Define the bounds for the coordinates, constraining them to the unit square [0, 1] x [0, 1]
    bounds = [(0, 1)] * (n * d)

    # Create a high-quality initial guess: a 4x4 grid.
    # This guides the optimizer towards a promising region of the search space,
    # dramatically improving performance over a random start.
    grid_side = int(np.sqrt(n))
    grid_coords = (np.arange(grid_side) + 0.5) / grid_side
    x_coords, y_coords = np.meshgrid(grid_coords, grid_coords)
    x0 = np.vstack([x_coords.ravel(), y_coords.ravel()]).T.ravel()

    # Use Dual Annealing, a powerful global optimization algorithm well-suited
    # for non-smooth, multi-modal objective functions.
    # Crucially, we specify a gradient-free local minimizer for dual_annealing's
    # internal local search, as the objective function is non-differentiable.
    # The default 'L-BFGS-B' would perform poorly.
    minimizer_kwargs = {"method": "Nelder-Mead"} # Nelder-Mead is robust for non-smooth functions

    result = dual_annealing(
        func=objective_func,
        bounds=bounds,
        args=(n, d),
        x0=x0,          # Provide the structured initial guess
        maxiter=10000,  # Increased iterations for a more thorough global search
        seed=seed,
        minimizer_kwargs=minimizer_kwargs # Apply the gradient-free local minimizer
    )
    
    # Reshape the optimized variables back into point coordinates
    optimal_points = result.x.reshape(n, d)

    return optimal_points
# EVOLVE-BLOCK-END