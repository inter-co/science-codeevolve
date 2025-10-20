# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
from scipy.optimize import basinhopping, NonlinearConstraint

# Constants for the problem
N_CIRCLES = 32
EPSILON = 1e-9 # Small epsilon for floating-point comparisons

# --- Helper functions for constraint-based optimization ---

def _get_circles_from_params(params: np.ndarray) -> np.ndarray:
    """Reshape flattened parameters [xs, ys, rs] into (N, 3) circles array."""
    n = N_CIRCLES
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    return np.stack((x, y, r), axis=-1)

def _local_objective(params: np.ndarray) -> float:
    """Objective function for the local optimizer: minimize -sum(radii)."""
    # Radii are the last N_CIRCLES elements
    return -np.sum(params[2*N_CIRCLES:])

def _all_constraints_vectorized(params: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array of all inequality constraint values, where g(x) >= 0 is feasible.
    This is used by scipy.optimize.NonlinearConstraint.
    """
    circles = _get_circles_from_params(params)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    constraints = []

    # 1. Containment constraints (4*N constraints):
    # x - r >= 0  =>  x >= r
    constraints.append(x - r)
    # 1 - (x + r) >= 0  => x + r <= 1
    constraints.append(1 - (x + r))
    # y - r >= 0  => y >= r
    constraints.append(y - r)
    # 1 - (y + r) >= 0  => y + r <= 1
    constraints.append(1 - (y + r))

    # 2. Non-overlap constraints (N*(N-1)/2 constraints):
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # This is equivalent to: sqrt(dx^2 + dy^2) - (r_i + r_j) >= 0
    # To avoid sqrt, we use: dx^2 + dy^2 - (r_i + r_j)^2 >= 0
    idx1, idx2 = np.triu_indices(N_CIRCLES, k=1)
    
    dx = x[idx1] - x[idx2]
    dy = y[idx1] - y[idx2]
    dist_sq = dx**2 + dy**2
    
    sum_radii = r[idx1] + r[idx2]
    min_dist_sq = sum_radii**2
    
    constraints.append(dist_sq - min_dist_sq)
    
    return np.concatenate(constraints)

class CustomCircleStep:
    """A custom step function for basinhopping that intelligently perturbs circle parameters."""
    def __init__(self, n_circles: int, xy_stepsize: float = 0.06, r_stepsize: float = 0.0045):
        self.n_circles = n_circles
        self.xy_stepsize = xy_stepsize
        self.r_stepsize = r_stepsize

    def __call__(self, x_params: np.ndarray) -> np.ndarray:
        n = self.n_circles
        x_new = x_params.copy()
        
        # Perturb x and y coordinates (first 2*n elements)
        xy_perturb = np.random.uniform(-self.xy_stepsize, self.xy_stepsize, 2 * n)
        x_new[:2*n] += xy_perturb
        
        # Perturb radii (last n elements)
        r_perturb = np.random.uniform(-self.r_stepsize, self.r_stepsize, n)
        x_new[2*n:] += r_perturb
        
        # Ensure radii remain positive (bounds in the minimizer will also enforce this)
        x_new[2*n:] = np.maximum(x_new[2*n:], EPSILON) 
        return x_new

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii,
    using basinhopping with SLSQP and explicit non-linear constraints.
    """
    np.random.seed(42)
    n = N_CIRCLES

    # --- Initial Guess (Grid-based) ---
    grid_size = int(np.ceil(np.sqrt(n))) # 6x6 grid
    grid_points = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(grid_points, grid_points)
    initial_x = xx.flatten()[:n]
    initial_y = yy.flatten()[:n]
    # Perturb to break symmetry
    initial_x += (np.random.rand(n) - 0.5) / (grid_size * 5)
    initial_y += (np.random.rand(n) - 0.5) / (grid_size * 5)
    initial_r = np.full(n, 0.01) # Small initial radii
    initial_params = np.concatenate([initial_x, initial_y, initial_r])
    
    # --- Bounds and Constraints ---
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(EPSILON, 0.5)] * n
    nonlinear_constraint = NonlinearConstraint(_all_constraints_vectorized, 0, np.inf)

    # --- Optimizer Configuration ---
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'constraints': [nonlinear_constraint],
        'options': {'maxiter': 500, 'ftol': 1e-8}
    }
    
    take_step = CustomCircleStep(n_circles=n)

    # --- Run Global Optimization ---
    result = basinhopping(
        _local_objective,
        initial_params,
        minimizer_kwargs=minimizer_kwargs,
        niter=150,
        T=1.0,
        take_step=take_step,
        seed=42,
        disp=False
    )

    optimized_params = result.x
    optimized_circles = _get_circles_from_params(optimized_params)

    # Post-process to ensure strict feasibility due to potential float precision issues
    # Shrink radii slightly until all constraints are met with a small tolerance
    for _ in range(100):
        violations = _all_constraints_vectorized(optimized_params)
        if np.all(violations >= -1e-7):
            break
        optimized_params[2*n:] *= 0.9999
    
    optimized_circles = _get_circles_from_params(optimized_params)

    return optimized_circles

# EVOLVE-BLOCK-END
