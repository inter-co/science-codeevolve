# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint
from scipy.spatial.distance import pdist, squareform # New import for vectorized distance
import time
import sys # New import for the callback function

# Global variables for time-based callback (from Inspiration 1)
_START_TIME = 0.0
_MAX_EVAL_TIME = 240 # seconds, set to ensure reasonable runtime, allows for more iteration if it converges fast

# Constants (moved to global scope to avoid pickling issues with multiprocessing)
N_CIRCLES = 21
MIN_RADIUS = 1e-6 # Smallest allowed radius and tolerance for positive radii constraint. (Increased for numerical stability, from Insp. 1/3)
OVERLAP_TOLERANCE = 1e-8 # Tolerance for overlap checks (retained for final validation only, consistent with best inspirations).
RANDOM_SEED = 42 # For reproducibility of stochastic methods.

# Penalty weights (tuned for strong enforcement, inspired by Inspiration 2)
W_OVERLAP = 1e7 # Very high penalty for critical overlaps (Increased to further discourage overlaps)
W_BOUNDS = 1e5  # High penalty for boundary violations
W_NEGATIVE_RADIUS = 1e7 # Highest penalty for non-positive radii

# Differential Evolution parameters
DE_MAXITER = 5000 # Further increased iterations for global search to push for higher precision.
DE_POP_SIZE_MULTIPLIER = 10 # A common heuristic is popsize = 10 * D
DE_TOL = 0.0001 # Made tolerance more stringent for a finer global search.

# Local Refinement (SLSQP) parameters
SLSQP_MAXITER = 5000 # Further increased Max iterations for local refinement for higher precision.
SLSQP_FTOL = 1e-10 # Made function tolerance more stringent for a finer local refinement.

# Validation tolerance
VALIDATION_TOLERANCE = 1e-7 # Made validation tolerance stricter to match optimization precision.

# Top-level callback function for differential_evolution to enable time-based stopping (From Inspiration 1)
def _callback_function(xk, convergence):
    """
    Callback function to stop differential_evolution if time limit is exceeded.
    This function must be top-level for multiprocessing compatibility.
    """
    if (time.time() - _START_TIME) > _MAX_EVAL_TIME:
        sys.stdout.write(f"\nCallback: Stopping optimization due to time limit ({_MAX_EVAL_TIME}s).\n")
        sys.stdout.flush() # Ensure message is printed immediately
        return True # Return True to terminate optimization
    return False

# ----------------------------------------------------------------------
# Helper functions (moved to global scope)
# ----------------------------------------------------------------------

def _unpack_params(params: np.ndarray):
    """
    Unpacks the flattened parameter array into rectangle dimensions and circle data.
    params: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    """
    rect_width = params[0]
    # Clipping is removed; optimizer bounds are the source of truth (inspired by Insp. 2)
    rect_height = 2.0 - rect_width
    
    circle_data = params[1:].reshape(N_CIRCLES, 3)
    x = circle_data[:, 0]
    y = circle_data[:, 1]
    r = circle_data[:, 2]
    return rect_width, rect_height, x, y, r, circle_data

def _calculate_overlap_penalty(circles_data: np.ndarray) -> float:
    """
    Calculates the penalty for overlapping circles using squared violations (inspired by Insp. 2).
    circles_data: (N_CIRCLES, 3) array of (x, y, r)
    """
    num_circles = circles_data.shape[0]
    
    centers = circles_data[:, :2]
    radii = circles_data[:, 2]

    centers = circles_data[:, :2] # Extract (x, y) coordinates
    radii = circles_data[:, 2]

    if num_circles < 2: # No overlaps if 0 or 1 circles
        return 0.0

    dist_matrix_sq = squareform(pdist(centers, metric='sqeuclidean')) # Squared Euclidean distances
    radii_sum_matrix = radii[:, None] + radii[None, :] # Pairwise sum of radii
    min_dist_sq_matrix = radii_sum_matrix**2

    # violations[i, j] = (r_i + r_j)^2 - ((x_i - x_j)^2 + (y_i - y_j)^2)
    violations = min_dist_sq_matrix - dist_matrix_sq
    
    # Extract upper triangle (excluding diagonal) for unique pairs
    upper_triangle_indices = np.triu_indices(num_circles, k=1)
    
    # Return sum of squared positive violations (overlaps)
    return np.sum(np.maximum(0, violations[upper_triangle_indices])**2)

def _calculate_boundary_penalty(circles_data: np.ndarray, rect_width: float, rect_height: float) -> float:
    """
    Calculates the penalty for circles going out of bounds using squared violations (inspired by Insp. 2).
    circles_data: (N_CIRCLES, 3) array of (x, y, r)
    """
    x, y, r = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]

    # Use squared violations for a smoother penalty landscape
    penalty = np.sum(np.maximum(0, r - x)**2) # x - r < 0
    penalty += np.sum(np.maximum(0, x + r - rect_width)**2) # x + r > rect_width
    penalty += np.sum(np.maximum(0, r - y)**2) # y - r < 0
    penalty += np.sum(np.maximum(0, y + r - rect_height)**2) # y + r > rect_height

    return penalty

def _calculate_radius_penalty(radii: np.ndarray) -> float:
    """
    Calculates the penalty for non-positive radii using squared violations (inspired by Insp. 2).
    radii: (N_CIRCLES,) array of radii
    """
    # Ensure all radii are at least MIN_RADIUS
    # Use squared violations for a smoother penalty landscape
    penalty = np.sum(np.maximum(0, MIN_RADIUS - radii)**2)
    return penalty

def _objective_function_with_penalties(params: np.ndarray) -> float:
    """
    Objective function for global optimization (Differential Evolution).
    Minimizes -sum_radii + total_penalty.
    params: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    """
    rect_width, rect_height, _, _, radii, circles_data = _unpack_params(params)

    # Add a large penalty if rect_width or rect_height are degenerate
    degenerate_penalty = 0.0
    if rect_width < MIN_RADIUS * 2 or rect_height < MIN_RADIUS * 2: # Rectangle too small for even two circles
        degenerate_penalty = W_BOUNDS * 1000 

    overlap_penalty = _calculate_overlap_penalty(circles_data)
    boundary_penalty = _calculate_boundary_penalty(circles_data, rect_width, rect_height)
    radius_penalty = _calculate_radius_penalty(radii)

    total_penalty = (W_OVERLAP * overlap_penalty +
                     W_BOUNDS * boundary_penalty +
                     W_NEGATIVE_RADIUS * radius_penalty +
                     degenerate_penalty)
    
    sum_radii = np.sum(radii)
    objective = -sum_radii + total_penalty
    return objective

# Non-linear constraint function for local optimization (minimize, SLSQP)
# All constraints must be of the form `c(x) >= 0` and return a 1D array.
def _nonlinear_constraints_func(params: np.ndarray) -> np.ndarray:
    """
    Returns an array of constraint values, all of which must be >= 0 for a valid solution.
    params: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    """
    rect_width, rect_height, x, y, r, _ = _unpack_params(params)
    
    # 1. Positive Radii: r_i - MIN_RADIUS >= 0
    positive_radii_constraints = r - MIN_RADIUS
    
    # 2. Containment:
    containment_constraints = np.concatenate([
        x - r,                  # x_i >= r_i
        rect_width - x - r,     # rect_width - x_i >= r_i
        y - r,                  # y_i >= r_i
        rect_height - y - r     # rect_height - y_i >= r_i
    ])
    
    # 3. Non-overlapping (vectorized): dist_sq - min_dist_sq >= 0
    non_overlap_constraints = np.array([])
    if N_CIRCLES > 1:
        centers = np.column_stack((x, y))
        dist_sq_matrix = squareform(pdist(centers, metric='sqeuclidean'))
        
        radii_sum_matrix = r[:, None] + r[None, :]
        min_dist_sq_matrix = radii_sum_matrix**2

        upper_tri_indices = np.triu_indices(N_CIRCLES, k=1)
        
        # We need dist_sq_matrix - min_dist_sq_matrix >= 0
        non_overlap_constraints = dist_sq_matrix[upper_tri_indices] - min_dist_sq_matrix[upper_tri_indices]
            
    return np.concatenate([positive_radii_constraints, containment_constraints, non_overlap_constraints])

def _is_valid_configuration(params: np.ndarray, tolerance: float = VALIDATION_TOLERANCE) -> bool:
    """
    Checks if a configuration is strictly valid according to all constraints.
    """
    rect_width, rect_height, x, y, r, _ = _unpack_params(params)

    # Check W, H validity based on MIN_RADIUS
    if not (MIN_RADIUS <= rect_width <= 2.0 - MIN_RADIUS):
        return False
    # rect_height is derived as 2.0 - rect_width, so W+H=2.0 is always true by definition in floating point.
    # The bounds on rect_width ensure rect_height is also valid.

    # 1. Positive radii
    if np.any(r < MIN_RADIUS - tolerance):
        return False

    # 2. Containment
    if np.any(x - r < -tolerance) or \
       np.any(rect_width - x - r < -tolerance) or \
       np.any(y - r < -tolerance) or \
       np.any(rect_height - y - r < -tolerance):
        return False
    
    # 3. Non-overlapping (vectorized)
    if N_CIRCLES > 1:
        i_indices, j_indices = np.triu_indices(N_CIRCLES, k=1)
        
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[i_indices] + r[j_indices]
        min_dist_sq = r_sum**2
        
        if np.any(dist_sq < min_dist_sq - tolerance):
            return False
                
    return True

# ----------------------------------------------------------------------
# Main optimization function
# ----------------------------------------------------------------------

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4
    in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    global _START_TIME # Declare global to modify it for the callback
    _START_TIME = time.time() # Initialize global start time for the callback
    np.random.seed(RANDOM_SEED) # Set seed for reproducibility

    # ----------------------------------------------------------------------
    # 1. Global Search using Differential Evolution
    # ----------------------------------------------------------------------
    # Decision variables: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    # Total variables: 1 (rect_width) + N_CIRCLES * 3 = 1 + 21 * 3 = 64
    D = 1 + N_CIRCLES * 3

    # Define bounds for each variable for DE
    # Widen bounds to allow more extreme aspect ratios (inspired by Insp. 1 & 2)
    width_bounds = (MIN_RADIUS, 2.0 - MIN_RADIUS)
    x_y_max_bound = 2.0 # Max possible coordinate if rectangle is degenerate (e.g., 2x0)
    r_max_bound = 0.5 # Max radius for a single circle in a 1x1 square is 0.5.

    circle_bounds = []
    for _ in range(N_CIRCLES):
        circle_bounds.append((0.0, x_y_max_bound)) # x_i
        circle_bounds.append((0.0, x_y_max_bound)) # y_i
        circle_bounds.append((MIN_RADIUS, r_max_bound)) # r_i (must be positive)

    # Combine all bounds: [rect_width_bounds] + [x_i, y_i, r_i bounds for all circles]
    bounds = [width_bounds] + circle_bounds

    # Generate an initial population with a heuristic (inspired by Inspiration 1/3)
    popsize = DE_POP_SIZE_MULTIPLIER * D
    initial_population = np.zeros((popsize, D))

    for i in range(popsize):
        # Initialize rect_width: sample around 1.0 (square)
        initial_population[i, 0] = np.random.uniform(0.8, 1.2) 
        initial_population[i, 0] = np.clip(initial_population[i, 0], bounds[0][0], bounds[0][1])

        current_width = initial_population[i, 0]
        current_height = 2.0 - current_width

        # Dynamically determine grid dimensions for N_CIRCLES based on aspect ratio
        num_cols = int(np.ceil(np.sqrt(N_CIRCLES * current_width / current_height)))
        num_rows = int(np.ceil(N_CIRCLES / num_cols))
        
        safe_width = max(current_width, MIN_RADIUS * 100)
        safe_height = max(current_height, MIN_RADIUS * 100)
        cell_width = safe_width / num_cols
        cell_height = safe_height / num_rows
        
        # Use a slightly larger initial radius heuristic (inspired by all inspiration programs)
        avg_initial_r = np.clip(min(cell_width, cell_height) / 2.5, MIN_RADIUS, r_max_bound)

        # Initialize circles (x, y, r) using a jittered grid
        for j in range(N_CIRCLES):
            idx_x = 1 + j * 3
            idx_y = 1 + j * 3 + 1
            idx_r = 1 + j * 3 + 2

            row = j // num_cols
            col = j % num_cols
            
            base_x = col * cell_width + cell_width / 2.0
            base_y = row * cell_height + cell_height / 2.0

            jitter_x = np.random.uniform(-cell_width / 4.0, cell_width / 4.0)
            jitter_y = np.random.uniform(-cell_height / 4.0, cell_height / 4.0)

            initial_r = np.random.uniform(MIN_RADIUS, avg_initial_r) 
            initial_r = np.clip(initial_r, MIN_RADIUS, r_max_bound)
            
            prop_x = base_x + jitter_x
            prop_y = base_y + jitter_y
            
            # Clamp proposed x, y to be within the rectangle and respect the initial radius
            prop_x = np.clip(prop_x, initial_r, current_width - initial_r)
            prop_y = np.clip(prop_y, initial_r, current_height - initial_r)
            
            # Assign to initial population, clipping to overall optimization bounds for safety
            initial_population[i, idx_x] = np.clip(prop_x, bounds[idx_x][0], bounds[idx_x][1])
            initial_population[i, idx_y] = np.clip(prop_y, bounds[idx_y][0], bounds[idx_y][1])
            initial_population[i, idx_r] = initial_r

    print(f"Starting Differential Evolution global search ({DE_MAXITER} iterations, popsize {popsize})...")
    de_result = differential_evolution(
        func=_objective_function_with_penalties,
        bounds=bounds,
        strategy='randtobest1exp', # Good general-purpose strategy
        maxiter=DE_MAXITER,
        popsize=popsize,
        tol=DE_TOL,
        seed=RANDOM_SEED,
        disp=False,
        workers=-1, # Use all available CPU cores for parallel evaluation
        init=initial_population, # Provide the heuristically generated initial population
        polish=True, # Apply local optimization to DE's best result (inspired by Insp. 1 & 2)
        callback=_callback_function # Add the time-based callback for early stopping (from Inspiration 1)
    )
    print(f"Differential Evolution finished in {time.time() - _START_TIME:.2f} seconds.") # Use global start time
    print(f"DE best objective (penalized): {de_result.fun:.6f}, sum_radii: {-de_result.fun if de_result.fun < 0 else 0:.6f}")
    
    initial_guess_local = de_result.x

    # ----------------------------------------------------------------------
    # 2. Local Refinement using SLSQP
    # ----------------------------------------------------------------------
    print("Starting local refinement with SLSQP...")
    
    # Objective for local refinement is purely -sum(radii)
    objective_local = lambda p: -np.sum(_unpack_params(p)[4])

    # Bounds for local optimization (can be slightly wider than DE bounds if needed, but keep consistent)
    bounds_local_W = (MIN_RADIUS, 2.0 - MIN_RADIUS) 
    bounds_local_coords = (0.0, 2.0)
    bounds_local_radii = (MIN_RADIUS, 1.0) 

    param_bounds_local = [bounds_local_W] + [bounds_local_coords] * (N_CIRCLES * 2) + [bounds_local_radii] * N_CIRCLES
    
    # Convert to scipy.optimize.Bounds object
    lb = [b[0] for b in param_bounds_local]
    ub = [b[1] for b in param_bounds_local]
    bounds_obj = Bounds(lb, ub)

    # Define the non-linear constraints for SLSQP.
    slsqp_constraints = NonlinearConstraint(
        _nonlinear_constraints_func,
        lb=0, # All constraints must be >= 0
        ub=np.inf # No upper bound
    )

    local_result = minimize(
        fun=objective_local,
        x0=initial_guess_local,
        method='SLSQP',
        bounds=bounds_obj,
        constraints=[slsqp_constraints],
        options={'disp': False, 'maxiter': SLSQP_MAXITER, 'ftol': SLSQP_FTOL},
    )
    
    print(f"Local refinement finished in {time.time() - _START_TIME:.2f} seconds total.") # Use global start time
    print(f"Local best objective (negative sum radii): {local_result.fun:.6f}")

    # ----------------------------------------------------------------------
    # 3. Final Result Processing and Validation
    # ----------------------------------------------------------------------
    final_params = local_result.x
    
    # Validate the local result. If not valid, fall back to DE result.
    if not _is_valid_configuration(final_params, tolerance=VALIDATION_TOLERANCE):
        print("WARNING: Final configuration from local refinement is not strictly valid. Attempting to use DE result.")
        final_params = de_result.x
        if not _is_valid_configuration(final_params, tolerance=VALIDATION_TOLERANCE):
            print("ERROR: DE result is also not strictly valid. Returning the best available (potentially invalid) solution.")
            # As a last resort, ensure radii are positive even if other constraints fail
            _, _, _, _, r_fallback, _ = _unpack_params(final_params)
            r_fallback = np.maximum(r_fallback, MIN_RADIUS)
            # Reconstruct params to include the clipped radii, preserving W, x, y
            final_params = np.concatenate(([final_params[0]], final_params[1:1+N_CIRCLES*2], r_fallback))
            
    final_rect_width, final_rect_height, _, _, final_radii, final_circles_data = _unpack_params(final_params)
    
    # Explicitly ensure all radii are at least MIN_RADIUS in the output array for robustness
    final_circles_data[:, 2] = np.maximum(final_radii, MIN_RADIUS)
    
    # One final check on the adjusted circles_array before returning
    final_params_adjusted = np.concatenate(([final_rect_width], final_circles_data.flatten()))
    if not _is_valid_configuration(final_params_adjusted, tolerance=VALIDATION_TOLERANCE):
        print("WARNING: Final adjusted configuration is still not strictly valid. Returning it with caution.")

    print(f"Optimal Rectangle Dimensions: Width={final_rect_width:.8f}, Height={final_rect_height:.8f}")
    print(f"Maximized Sum of Radii: {np.sum(final_radii):.8f}")

    return final_circles_data

# EVOLVE-BLOCK-END

if __name__ == '__main__':
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
