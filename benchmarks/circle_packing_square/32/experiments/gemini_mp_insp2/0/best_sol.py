# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping, NonlinearConstraint # Added NonlinearConstraint for improved constraint handling
import time

# --- Constants ---
N_CIRCLES = 32
RANDOM_SEED = 42
# N_STARTS is no longer directly used for manual multi-start, basinhopping's niter replaces it

# --- Helper Functions for Optimization ---

def _get_circle_params(p: np.ndarray) -> np.ndarray:
    """Reshapes the 1D parameter array p into a (N_CIRCLES, 3) matrix."""
    return p.reshape(N_CIRCLES, 3)

def objective(p: np.ndarray) -> float:
    """Objective function: negative sum of radii (for minimization)."""
    radii = p[2::3] # Select every third element starting from index 2 (r1, r2, ...)
    return -np.sum(radii)

# Combined and vectorized constraint function adapted from Inspiration 1
def _all_constraints_vectorized(p: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array of all inequality constraint values, where g(x) >= 0 is feasible.
    This is used by scipy.optimize.NonlinearConstraint.
    """
    circles = _get_circle_params(p) # Use existing helper to get (N, 3) array
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
    # To avoid sqrt during intermediate calculations, we use: dx^2 + dy^2 - (r_i + r_j)^2 >= 0
    idx1, idx2 = np.triu_indices(N_CIRCLES, k=1)
    
    dx = x[idx1] - x[idx2]
    dy = y[idx1] - y[idx2]
    dist_sq = dx**2 + dy**2
    
    sum_radii = r[idx1] + r[idx2]
    min_dist_sq = sum_radii**2
    
    constraints.append(dist_sq - min_dist_sq)
    
    return np.concatenate(constraints)

# Custom step function adapted from Inspiration 1, for the target's parameter array format.
class CustomCircleStep:
    """A custom step function for basinhopping that intelligently perturbs circle parameters."""
    def __init__(self, n_circles: int, xy_stepsize: float = 0.05, r_stepsize: float = 0.004):
        self.n_circles = n_circles
        self.xy_stepsize = xy_stepsize
        self.r_stepsize = r_stepsize

    def __call__(self, x_params: np.ndarray) -> np.ndarray:
        """Vectorized perturbation for the [x1, y1, r1, ...] parameter structure."""
        n = self.n_circles
        x_new = x_params.copy()
        
        # Generate random perturbations for all coordinates at once
        x_perturb = np.random.uniform(-self.xy_stepsize, self.xy_stepsize, n)
        y_perturb = np.random.uniform(-self.xy_stepsize, self.xy_stepsize, n)
        r_perturb = np.random.uniform(-self.r_stepsize, self.r_stepsize, n)

        # Apply perturbations using slicing for efficiency
        x_new[0::3] += x_perturb
        x_new[1::3] += y_perturb
        x_new[2::3] += r_perturb
        
        # Ensure radii remain positive
        x_new[2::3] = np.maximum(x_new[2::3], 1e-7) 
        return x_new

def generate_grid_initial_guess(rows: int, cols: int, seed: int = None, initial_r: float = 0.01) -> np.ndarray:
    """
    Generates a grid-based initial guess. This version is vectorized and uses a
    perturbation scale inspired by the successful inspiration programs.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create grid points
    x_coords = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols)
    y_coords = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    initial_x = xx.flatten()
    initial_y = yy.flatten()

    # Define perturbation scale based on the smaller grid spacing, similar to inspirations
    min_spacing = min(1.0 / cols, 1.0 / rows)
    # Perturb by up to 10% of min spacing (since rand-0.5 is in [-0.5, 0.5])
    perturb_scale = min_spacing / 5.0 
    
    # Apply vectorized perturbation
    initial_x += (np.random.rand(N_CIRCLES) - 0.5) * perturb_scale
    initial_y += (np.random.rand(N_CIRCLES) - 0.5) * perturb_scale

    # Assemble the parameter vector [x1, y1, r1, ...]
    p0 = np.zeros(N_CIRCLES * 3)
    p0[0::3] = initial_x
    p0[1::3] = initial_y
    p0[2::3] = initial_r

    return p0


# Removed validate_solution as post-processing loop will handle final validation and adjustment.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses scipy.optimize.basinhopping with SLSQP as the local minimizer.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Define bounds for x, y, r for each circle
    # x_i in [0, 1], y_i in [0, 1], r_i in [1e-6, 0.5]
    # Corrected bounds definition for the target's [x1,y1,r1, ...] parameter array format
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((1e-6, 0.5)) # r_i (lower bound 1e-6 for numerical stability, max 0.5)

    # Define the single NonlinearConstraint object using the vectorized function
    nonlinear_constraint = NonlinearConstraint(_all_constraints_vectorized, 0, np.inf)

    start_time = time.time()

    # --- Use basinhopping for global optimization ---
    # Configure the local minimizer (SLSQP), tuned for speed based on Inspiration 2
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'constraints': [nonlinear_constraint], # Use the single NonlinearConstraint
        'options': {'disp': False, 'maxiter': 1500, 'ftol': 1e-9} # Tuned for speed
    }
    
    # Initial guess for basinhopping, using the improved vectorized generator.
    p0 = generate_grid_initial_guess(rows=4, cols=8, seed=RANDOM_SEED, initial_r=0.01)
    
    # Ensure initial guess satisfies bounds by clipping, as perturbations might push points slightly out of [0,1].
    p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Instantiate the custom step function, now fully vectorized.
    take_step = CustomCircleStep(n_circles=N_CIRCLES, xy_stepsize=0.05, r_stepsize=0.004)

    # Run basinhopping, with parameters tuned to balance runtime and solution quality.
    res = basinhopping(
        objective,
        p0,
        minimizer_kwargs=minimizer_kwargs,
        niter=150,     # Reduced global iterations to balance speed and quality
        T=1.2,         # Temperature parameter from inspirations
        take_step=take_step, # Use the custom step function
        seed=RANDOM_SEED, # For reproducibility
        disp=False     # Set to True for verbose output
    )

    eval_time = time.time() - start_time

    optimized_params = res.x
    
    # Post-process to ensure strict feasibility due to potential float precision issues, as in Inspiration 1
    # Shrink radii slightly until all constraints are met with a small tolerance
    # Radii are at p[2::3] in the flattened parameter array
    for _ in range(100): # Max 100 small adjustments
        violations = _all_constraints_vectorized(optimized_params)
        if np.all(violations >= -1e-7): # Check against 1e-7 tolerance
            break
        optimized_params[2::3] *= 0.9999 # Shrink all radii by a tiny fraction

    # Reshape the best parameter array into (N_CIRCLES, 3) (x, y, r)
    circles_result = _get_circle_params(optimized_params)

    # A final clamp for radii to ensure non-negativity after post-processing,
    # though bounds and shrinking should largely handle this.
    circles_result[:, 2] = np.maximum(0, circles_result[:, 2])

    return circles_result

# EVOLVE-BLOCK-END
