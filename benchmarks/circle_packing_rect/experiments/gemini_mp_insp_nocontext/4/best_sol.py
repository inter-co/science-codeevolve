# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint
import time

# --- Constants (tuned based on Inspiration Program 2) ---
N_CIRCLES = 21
MIN_RADIUS = 1e-6 # Smallest allowed radius and tolerance for positive radii constraint.
OVERLAP_TOLERANCE = 1e-9 # Tolerance for overlap checks (enforces slightly larger gaps).
RANDOM_SEED = 42 # For reproducibility of stochastic methods.

# Penalty weights (tuned for strong enforcement, from Inspiration 2)
W_OVERLAP = 1e7 # Very high penalty for critical overlaps
W_BOUNDS = 1e5  # High penalty for boundary violations
W_NEGATIVE_RADIUS = 1e7 # Highest penalty for non-positive radii

# Differential Evolution parameters (tuned from Inspiration 2)
DE_MAXITER = 4000 # Increased iterations for global search
DE_POP_SIZE_MULTIPLIER = 10 # A common heuristic is popsize = 10 * D
DE_TOL = 0.001 # Tolerance for DE convergence

# Local Refinement (SLSQP) parameters (tuned from Inspiration 2)
SLSQP_MAXITER = 4000 # Increased Max iterations for local refinement
SLSQP_FTOL = 1e-9 # Function tolerance for local refinement

# Validation tolerance
VALIDATION_TOLERANCE = 1e-6 # A slightly more lenient tolerance for final validity check

# ----------------------------------------------------------------------
# Helper functions (moved to global scope for multiprocessing compatibility)
# ----------------------------------------------------------------------

def _unpack_params(params: np.ndarray):
    """
    Unpacks the flattened parameter array into rectangle dimensions and circle data.
    params: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    """
    rect_width = params[0]
    rect_height = 2.0 - rect_width
    
    circle_data = params[1:].reshape(N_CIRCLES, 3)
    x = circle_data[:, 0]
    y = circle_data[:, 1]
    r = circle_data[:, 2]
    return rect_width, rect_height, x, y, r, circle_data

def _calculate_overlap_penalty(circles_data: np.ndarray) -> float:
    """
    Calculates the penalty for overlapping circles using squared violations.
    circles_data: (N_CIRCLES, 3) array of (x, y, r)
    """
    num_circles = circles_data.shape[0]
    
    centers = circles_data[:, :2]
    radii = circles_data[:, 2]

    # Calculate squared distances between all circle centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq_matrix = np.sum(diff**2, axis=2)

    # Calculate squared sum of radii for all pairs
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq_matrix = radii_sum_matrix**2

    # violations[i, j] = (r_i + r_j)^2 - ((x_i - x_j)^2 + (y_i - y_j)^2)
    # Add OVERLAP_TOLERANCE to make violations more sensitive, enforcing slightly larger gaps
    violations = min_dist_sq_matrix - dist_sq_matrix + OVERLAP_TOLERANCE
    
    violations[np.diag_indices(num_circles)] = 0 # Exclude self-comparison
    violations[np.tril_indices(num_circles)] = 0 # Exclude lower triangle to avoid double counting

    # Return sum of squared positive violations for a smoother penalty landscape
    return np.sum(np.maximum(0, violations)**2)

def _calculate_boundary_penalty(circles_data: np.ndarray, rect_width: float, rect_height: float) -> float:
    """
    Calculates the penalty for circles going out of bounds using squared violations.
    circles_data: (N_CIRCLES, 3) array of (x, y, r)
    """
    x, y, r = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]

    penalty = np.sum(np.maximum(0, r - x)**2) # x - r < 0
    penalty += np.sum(np.maximum(0, x + r - rect_width)**2) # x + r > rect_width
    penalty += np.sum(np.maximum(0, r - y)**2) # y - r < 0
    penalty += np.sum(np.maximum(0, y + r - rect_height)**2) # y + r > rect_height

    return penalty

def _calculate_radius_penalty(radii: np.ndarray) -> float:
    """
    Calculates the penalty for non-positive radii using squared violations.
    radii: (N_CIRCLES,) array of radii
    """
    # Ensure all radii are at least MIN_RADIUS
    penalty = np.sum(np.maximum(0, MIN_RADIUS - radii)**2)
    return penalty

def _objective_function_with_penalties(params: np.ndarray) -> float:
    """
    Objective function for global optimization (Differential Evolution).
    Minimizes -sum_radii + total_penalty.
    Uses quadratic penalties for a smoother optimization landscape.
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

# Objective for local refinement is purely -sum(radii)
def _objective_local_refinement(params: np.ndarray) -> float:
    return -np.sum(_unpack_params(params)[4])

# Non-linear constraint function for local optimization (minimize, SLSQP)
# All constraints must be of the form `c(x) >= 0` and return a 1D array.
def _nonlinear_constraints_func(params: np.ndarray) -> np.ndarray:
    """
    Returns an array of constraint values, all of which must be >= 0 for a valid solution.
    params: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    """
    rect_width, rect_height, x, y, r, _ = _unpack_params(params)
    
    constraints = []
    
    # 1. Positive Radii: r_i - MIN_RADIUS >= 0
    constraints.extend(r - MIN_RADIUS)
    
    # 2. Containment:
    constraints.extend(x - r)          # x_i >= r_i
    constraints.extend(rect_width - x - r)      # rect_width - x_i >= r_i
    constraints.extend(y - r)          # y_i >= r_i
    constraints.extend(rect_height - y - r)      # rect_height - y_i >= r_i
    
    # 3. Non-overlapping (vectorized): dist_sq - min_dist_sq >= 0
    if N_CIRCLES > 1:
        i_indices, j_indices = np.triu_indices(N_CIRCLES, k=1)
        
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[i_indices] + r[j_indices]
        min_dist_sq = r_sum**2
        
        # We need dist_sq - min_dist_sq >= OVERLAP_TOLERANCE
        constraints.extend(dist_sq - min_dist_sq - OVERLAP_TOLERANCE)
            
    return np.array(constraints)

def _is_valid_configuration(params: np.ndarray, tolerance: float = VALIDATION_TOLERANCE) -> bool:
    """
    Checks if a configuration is strictly valid according to all constraints.
    """
    rect_width, rect_height, x, y, r, _ = _unpack_params(params)

    # Check W, H validity based on MIN_RADIUS and perimeter
    # Adjusted lower bound for rect_width to MIN_RADIUS, consistent with optimizer bounds
    if not (MIN_RADIUS <= rect_width <= 2.0 - MIN_RADIUS and abs(rect_width + rect_height - 2.0) < tolerance):
        return False

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
    in order to maximize the sum of their radii, using a hybrid Differential Evolution
    and SLSQP approach inspired by the best-performing solution.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time_overall = time.time()
    np.random.seed(RANDOM_SEED) # Set seed for reproducibility

    # ----------------------------------------------------------------------
    # 1. Global Search using Differential Evolution
    # ----------------------------------------------------------------------
    # Decision variables: [rect_width, x1, y1, r1, ..., x21, y21, r21]
    # Total variables: 1 (rect_width) + N_CIRCLES * 3 = 1 + 21 * 3 = 64
    D = 1 + N_CIRCLES * 3

    # Define bounds for each variable for DE
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

    # Generate an initial population with a heuristic (from Inspiration 2)
    popsize = DE_POP_SIZE_MULTIPLIER * D # popsize = 10 * 64 = 640
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
        
        safe_width = max(current_width, MIN_RADIUS * 100) # Ensure no division by zero or very small numbers
        safe_height = max(current_height, MIN_RADIUS * 100)
        cell_width = safe_width / num_cols
        cell_height = safe_height / num_rows
        
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
        strategy='randtobest1exp', # Strategy from Inspiration 2
        maxiter=DE_MAXITER, 
        popsize=popsize,
        tol=DE_TOL,
        seed=RANDOM_SEED,
        disp=False, # Set to True for verbose output
        workers=-1, # Use all available CPU cores for parallel evaluation
        init=initial_population, # Provide the heuristically generated initial population
        polish=True # Apply local optimization to DE's best result (from Inspiration 2)
    )
    print(f"Differential Evolution finished in {time.time() - start_time_overall:.2f} seconds.")
    # For DE, the reported fun includes penalties, so we need to re-evaluate without penalties
    sum_radii_de = -_objective_local_refinement(de_result.x)
    print(f"DE best sum_radii (unpenalized): {sum_radii_de:.6f}")
    
    initial_guess_local = de_result.x

    # ----------------------------------------------------------------------
    # 2. Local Refinement using SLSQP
    # ----------------------------------------------------------------------
    print("Starting local refinement with SLSQP...")
    
    # Objective for local refinement is purely -sum(radii)
    # _objective_local_refinement function is already defined
    
    # Bounds for local optimization
    bounds_local_W = (MIN_RADIUS, 2.0 - MIN_RADIUS) 
    bounds_local_coords = (0.0, 2.0)
    bounds_local_radii = (MIN_RADIUS, 1.0) # Allow larger radii, as constraints will pull them back

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
        fun=_objective_local_refinement,
        x0=initial_guess_local,
        method='SLSQP', # Using SLSQP as in Inspiration 2
        bounds=bounds_obj,
        constraints=[slsqp_constraints],
        options={'disp': False, 'maxiter': SLSQP_MAXITER, 'ftol': SLSQP_FTOL}, 
    )
    
    print(f"Local refinement finished. Total time: {time.time() - start_time_overall:.2f} seconds.")
    print(f"Local best objective (negative sum radii): {local_result.fun:.6f}")

    # ----------------------------------------------------------------------
    # 3. Final Result Processing and Validation
    # ----------------------------------------------------------------------
    
    # Select the best valid solution
    final_params = None
    if local_result.success and _is_valid_configuration(local_result.x):
        final_params = local_result.x
        print("Local refinement produced a valid solution.")
    elif _is_valid_configuration(de_result.x):
        final_params = de_result.x
        print("Local refinement failed or was not better/valid. Using DE result.")
    else:
        print("ERROR: Neither local refinement nor DE produced a valid solution. Returning zeros.")
        return np.zeros((N_CIRCLES, 3))

    # If local_result was not successful but de_result was valid, use de_result
    if final_params is None:
        print("ERROR: No valid solution found after optimization stages.")
        return np.zeros((N_CIRCLES, 3))

    final_rect_width, final_rect_height, _, _, final_radii, final_circles_data = _unpack_params(final_params)
    
    # Explicitly ensure all radii are at least MIN_RADIUS in the output array for robustness
    final_circles_data[:, 2] = np.maximum(final_radii, MIN_RADIUS)
    
    # One final check on the adjusted circles_array before returning
    final_params_adjusted = np.concatenate(([final_rect_width], final_circles_data.flatten()))
    if not _is_valid_configuration(final_params_adjusted, tolerance=VALIDATION_TOLERANCE):
        print("WARNING: Final adjusted configuration is still not strictly valid. Returning it with caution.")

    sum_r_final = np.sum(final_circles_data[:, 2])
    print(f"\nOptimization complete.")
    print(f"  - Final Sum of radii: {sum_r_final:.12f}")
    benchmark_value = 2.3658321334167627
    print(f"  - Benchmark ratio: {sum_r_final / benchmark_value:.6f}")
    print(f"  - Optimized Rectangle (W, H): ({final_rect_width:.6f}, {final_rect_height:.6f})")
    print(f"  - Total evaluation time: {time.time() - start_time_overall:.2f} seconds")

    return final_circles_data

# EVOLVE-BLOCK-END

if __name__ == '__main__':
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
