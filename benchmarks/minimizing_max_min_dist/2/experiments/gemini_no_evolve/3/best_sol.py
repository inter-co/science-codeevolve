# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2D to maximize d_min/d_max ratio.
    This is solved by reformulating the problem as a constrained optimization:
    Maximize d_min subject to d_max <= 1.
    This is a 'maximin' problem, well-suited for SLSQP.

    1. An initial guess is generated from a perturbed 4x4 grid, a known
       strong starting configuration.
    2. SLSQP is used to solve the constrained problem:
       - Objective: maximize delta (a proxy for d_min)
       - Variables: 16*2 coordinates + delta
       - Constraints:
         - d_ij >= delta  (for all pairs i, j)
         - d_ij <= 1      (ensures d_max is bounded)
    3. The final point set is scaled and translated to fit the [0,1]x[0,1] box.
    """
    n_points = 16
    n_dims = 2
    n_vars_coords = n_points * n_dims

    # The state vector for the optimizer is x = [coord_1, ..., coord_32, delta].
    # We want to maximize delta, which is equivalent to minimizing -delta.
    def objective(x: np.ndarray) -> float:
        """Objective function: minimize -delta."""
        return -x[-1]

    # The constraints ensure that all pairwise distances d_ij satisfy:
    # delta <= d_ij <= 1.
    def constraints(x: np.ndarray) -> np.ndarray:
        """
        Returns an array of constraint violations.
        For SLSQP, constraints are of the form c(x) >= 0.
        """
        coords = x[:-1].reshape(n_points, n_dims)
        delta = x[-1]
        distances = pdist(coords)
        
        # Constraint 1: d_ij - delta >= 0  (ensures d_min >= delta)
        c1 = distances - delta
        # Constraint 2: 1.0 - d_ij >= 0     (ensures d_max <= 1)
        c2 = 1.0 - distances
        
        return np.concatenate((c1, c2))

    # --- Initial Guess ---
    # A good initial guess is crucial. We start with a 4x4 grid, which is
    # a strong candidate configuration, and perturb it slightly.
    grid_coords = np.linspace(0, 1, 4)
    x_grid, y_grid = np.meshgrid(grid_coords, grid_coords)
    initial_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    # Add small random noise to break perfect symmetry, which can help the
    # optimizer escape saddle points.
    rng = np.random.default_rng(42)
    initial_points += rng.uniform(-0.01, 0.01, size=initial_points.shape)
    
    # Scale the initial guess so its d_max is approximately 1, matching our constraint.
    # The d_max of a grid in [0,1]x[0,1] is sqrt(2).
    initial_points /= np.sqrt(2)

    # Initial delta guess: d_min of the scaled grid.
    # d_min of the original grid is 1/3. Scaled d_min is (1/3)/sqrt(2).
    initial_delta = (1/3) / np.sqrt(2)
    
    # Combine into the initial state vector for the optimizer.
    x0 = np.concatenate([initial_points.flatten(), [initial_delta]])

    # --- Bounds ---
    # Coordinates are unbounded during optimization; the d_max<=1 constraint
    # contains them. Delta (our d_min proxy) is bounded.
    bounds = [(None, None)] * n_vars_coords + [(1e-6, 1.0)]

    # --- Optimization ---
    # SLSQP is used as it's designed for this type of constrained problem.
    cons = {'type': 'ineq', 'fun': constraints}
    result = minimize(
        fun=objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
    )

    # --- Post-processing ---
    # Extract optimized coordinates and scale/translate to fit in the unit square.
    # This final transformation does not change the d_min/d_max ratio.
    optimized_points = result.x[:-1].reshape(n_points, n_dims)

    # Translate points so that the minimum coordinates are at (0, 0).
    optimized_points -= optimized_points.min(axis=0)
    
    # Scale points so that the maximum coordinate in any dimension is 1.
    # This fits the configuration tightly into the unit square.
    max_coord = optimized_points.max()
    if max_coord > 1e-9: # Avoid division by zero if all points are at origin
        optimized_points /= max_coord
        
    return optimized_points
# EVOLVE-BLOCK-END