# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial.distance import pdist, squareform
import time

# --- Constants for Optimization ---
N_CIRCLES = 26
MIN_RADIUS_ALLOWED = 1e-7 # Use a slightly smaller value for robustness with SLSQP
GLOBAL_OPTIMIZATION_SEED = 42 # For reproducibility of random processes

# --- Objective Function (to be minimized) ---
def objective_function(params):
    """
    The objective is to maximize the sum of radii.
    For a minimizer, we minimize the negative sum of radii.
    """
    # params are a flat array [x1, y1, r1, x2, y2, r2, ...]
    return -np.sum(params[2::3])

# --- Constraint Function for SLSQP ---
def all_constraints(params):
    """
    Returns a single array of values for all inequality constraints.
    For SLSQP, each value must be >= 0 for the constraint to be satisfied.
    This function is fully vectorized for performance.
    """
    circles = params.reshape((N_CIRCLES, 3))
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Containment constraints (4 * N_CIRCLES)
    # c >= 0  =>  x - r >= 0, (1 - x) - r >= 0, etc.
    containment_constraints = np.concatenate([
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ])

    # 2. Non-overlap constraints (N_CIRCLES * (N_CIRCLES - 1) / 2)
    # c >= 0  =>  (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    if N_CIRCLES > 1:
        centers = circles[:, :2]
        # pdist computes condensed pairwise squared Euclidean distances
        dist_sq = pdist(centers, 'sqeuclidean')

        # Compute condensed pairwise sum of radii squared
        # np.triu_indices gives the upper-triangle indices, matching pdist's order
        i_upper, j_upper = np.triu_indices(N_CIRCLES, k=1)
        radii_sum_sq = (r[i_upper] + r[j_upper])**2

        overlap_constraints = dist_sq - radii_sum_sq
        return np.concatenate([containment_constraints, overlap_constraints])
    
    return containment_constraints


# --- Initial Configuration Generation ---
def _generate_initial_circles(n_circles, rng):
    """
    Generates a structured initial configuration of circles on a jittered grid.
    This provides a better starting point than pure random placement.
    Introduces some variation in initial radii.
    """
    initial_params = np.zeros(n_circles * 3)
    
    # Place circles on a slightly jittered grid to start with a good distribution
    # For 26 circles, a 5x5 grid (25 circles) plus one extra point is a good start.
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    xs = np.linspace(0.1, 0.9, grid_size)
    ys = np.linspace(0.1, 0.9, grid_size)
    grid_points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    
    for i in range(n_circles):
        # Introduce some variation in initial radii
        r_init = rng.uniform(0.04, 0.06) # Slightly varied small initial radii
        
        # Get position from grid, with some randomness
        px, py = grid_points[i % len(grid_points)]
        x_init = px + rng.uniform(-0.05, 0.05)
        y_init = py + rng.uniform(-0.05, 0.05)
        
        # Ensure initial placement respects radius and square boundaries
        initial_params[i*3]     = np.clip(x_init, r_init, 1.0 - r_init)
        initial_params[i*3 + 1] = np.clip(y_init, r_init, 1.0 - r_init)
        initial_params[i*3 + 2] = r_init
        
    return initial_params

# --- Validation Framework ---
def validate_circles(circles, epsilon=1e-9):
    """
    Validates a set of circles for containment and non-overlap.
    circles: np.array of shape (N,3) where (x,y,r)
    Returns: (is_valid, sum_radii)
    """
    n = circles.shape[0]
    if n == 0:
        return False, 0.0

    sum_radii = np.sum(circles[:, 2])

    # 1. Positive Radii Check
    if not np.all(circles[:, 2] > 0 - epsilon): 
        return False, sum_radii

    # 2. Containment Check
    x_coords, y_coords, radii = circles[:, 0], circles[:, 1], circles[:, 2]
    if not (np.all(x_coords >= radii - epsilon) and
            np.all(x_coords <= 1 - radii + epsilon) and
            np.all(y_coords >= radii - epsilon) and
            np.all(y_coords <= 1 - radii + epsilon)):
        return False, sum_radii

    # 3. Non-overlap Check (vectorized)
    if n > 1:
        centers = circles[:, :2]
        dist_sq_matrix = squareform(pdist(centers, 'sqeuclidean'))
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        radii_sum_sq_matrix = radii_sum * radii_sum
        
        i_upper, j_upper = np.triu_indices(n, k=1)
        if np.any(dist_sq_matrix[i_upper, j_upper] < radii_sum_sq_matrix[i_upper, j_upper] - epsilon):
            return False, sum_radii

    return True, sum_radii


def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    This version uses a global optimization strategy (basinhopping) with a robust
    local minimizer (SLSQP) that handles constraints explicitly.
    """
    rng = np.random.default_rng(GLOBAL_OPTIMIZATION_SEED)
    
    # Generate a more structured initial configuration
    initial_params = _generate_initial_circles(N_CIRCLES, rng)

    # Define simple box bounds for x, y, r for the local minimizer (SLSQP).
    bounds_flat = []
    for _ in range(N_CIRCLES):
        bounds_flat.append((0.0, 1.0)) # x_i coordinate
        bounds_flat.append((0.0, 1.0)) # y_i coordinate
        bounds_flat.append((MIN_RADIUS_ALLOWED, 0.5)) # r_i radius

    # SLSQP is better suited for this problem as it handles constraints directly.
    # Define minimizer_kwargs for basinhopping's internal local minimizer.
    basinhopping_minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds_flat,
        "constraints": {'type': 'ineq', 'fun': all_constraints},
        "options": {
            "maxiter": 500, # Reduced maxiter for faster local convergence during global search
            "ftol": 1e-7,   # Slightly looser tolerance for faster convergence
            "disp": False
        }
    }
    
    # Basinhopping for global optimization.
    # Reduced niter to fit within time limits, adjusted T and stepsize.
    bh_result = basinhopping(
        func=objective_function,
        x0=initial_params,
        niter=100,       # Reduced global search iterations for speed
        T=1.5,           # Adjusted temperature for exploration
        stepsize=0.07,   # Adjusted step size
        minimizer_kwargs=basinhopping_minimizer_kwargs,
        disp=False,      # Do not print status messages
        seed=GLOBAL_OPTIMIZATION_SEED
    )

    # Reshape the optimized parameters back into (N, 3) circle format
    optimized_params = bh_result.x
    circles = optimized_params.reshape((N_CIRCLES, 3))
    
    # Clip radii just in case the optimizer slightly violates the bound
    circles[:, 2] = np.maximum(circles[:, 2], MIN_RADIUS_ALLOWED)

    # Validate the final solution and attempt a repair if needed
    is_valid, final_sum_radii = validate_circles(circles)
    if not is_valid:
        print(f"Warning: Basinhopping solution (sum_radii={final_sum_radii:.6f}) is invalid. Attempting final repair...")
        
        # Use a stricter, dedicated local minimizer for the repair step
        # This ensures the final solution is valid even if basinhopping's steps were slightly less precise.
        repair_minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds_flat,
            "constraints": {'type': 'ineq', 'fun': all_constraints},
            "options": {
                "maxiter": 3000, # Increased maxiter for thorough final convergence
                "ftol": 1e-9,    # Stricter tolerance for final result
                "disp": False
            }
        }
        repair_result = minimize(objective_function, optimized_params, **repair_minimizer_kwargs)
        
        if repair_result.success:
            circles = repair_result.x.reshape((N_CIRCLES, 3))
            is_valid_repair, final_sum_radii_repair = validate_circles(circles)
            if is_valid_repair:
                print(f"Repair successful. New sum_radii={final_sum_radii_repair:.6f}")
            else:
                 print(f"Repair step failed to produce a valid solution. Final sum_radii={final_sum_radii_repair:.6f}")
        else:
            print("Repair step optimizer failed to converge.")
    
    return circles

# EVOLVE-BLOCK-END