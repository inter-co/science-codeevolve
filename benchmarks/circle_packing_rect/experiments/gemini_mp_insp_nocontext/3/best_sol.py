# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from scipy.spatial.distance import pdist, squareform # New import for vectorized distance
import time
import sys

# Global Constants
N_CIRCLES = 21
MIN_RADIUS = 1e-6 # Smallest allowed radius and tolerance for positive radii constraint (consistent with Inspirations 2/3)
VALIDATION_TOLERANCE = 1e-6 # Tolerance for final validity check

# Penalty weights for DE objective function (tuned for strong enforcement, from Inspirations 2/3)
W_OVERLAP = 1e7 # Very high penalty for critical overlaps
W_BOUNDS = 1e5  # High penalty for boundary violations
W_NEGATIVE_RADIUS = 1e7 # Highest penalty for non-positive radii

# Global variables for time-based callback (from Inspiration 1)
_START_TIME = 0.0
_MAX_EVAL_TIME = 350 # seconds, leaving buffer for final processing and validation

# Helper Functions (moved to global scope for multiprocessing compatibility)

def _unpack_params(params: np.ndarray):
    """
    Unpacks the flattened parameter array into rectangle dimensions and circle data.
    params structure: [rect_width, x1, y1, r1, ..., xN, yN, rN] (Aligned with Inspirations 1/2/3)
    """
    rect_width = params[0]
    rect_height = 2.0 - rect_width # Perimeter constraint W + H = 2
    
    circle_data = params[1:].reshape(N_CIRCLES, 3)
    x_coords = circle_data[:, 0]
    y_coords = circle_data[:, 1]
    radii = circle_data[:, 2]
    return rect_width, rect_height, x_coords, y_coords, radii, circle_data

# Penalty calculation helper functions for DE objective (from Inspirations 2/3)
def _calculate_overlap_penalty(circles_data: np.ndarray) -> float:
    """
    Calculates the penalty for overlapping circles using squared violations.
    circles_data: (N_CIRCLES, 3) array of (x, y, r)
    Uses scipy.spatial.distance for efficiency (inspired by Inspiration 1).
    """
    num_circles = circles_data.shape[0]
    centers = circles_data[:, :2] # Extract (x, y) coordinates
    radii = circles_data[:, 2]

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
    Calculates the penalty for circles going out of bounds using squared violations.
    circles_data: (N_CIRCLES, 3) array of (x, y, r)
    """
    x, y, r = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]

    penalty = np.sum(np.maximum(0, r - x)**2)          # x_i - r_i < 0
    penalty += np.sum(np.maximum(0, x + r - rect_width)**2) # x_i + r_i > rect_width
    penalty += np.sum(np.maximum(0, r - y)**2)          # y_i - r_i < 0
    penalty += np.sum(np.maximum(0, y + r - rect_height)**2) # y_i + r_i > rect_height

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
    Uses quadratic penalties for a smoother optimization landscape. (From Inspirations 2/3)
    params: [rect_width, x1, y1, r1, ..., xN, yN, rN]
    """
    rect_width, rect_height, _, _, radii, circles_data = _unpack_params(params)

    # Objective: Maximize sum of radii -> Minimize negative sum of radii
    objective = -np.sum(radii)
    penalty = 0.0

    # Add a large penalty if rect_width or rect_height are degenerate (from Inspirations 2/3)
    if rect_width < MIN_RADIUS * 2 or rect_height < MIN_RADIUS * 2:
        penalty += W_BOUNDS * 1000 # High penalty for degenerate rectangle

    overlap_penalty = _calculate_overlap_penalty(circles_data)
    boundary_penalty = _calculate_boundary_penalty(circles_data, rect_width, rect_height)
    radius_penalty = _calculate_radius_penalty(radii)

    total_penalty = (W_OVERLAP * overlap_penalty +
                     W_BOUNDS * boundary_penalty +
                     W_NEGATIVE_RADIUS * radius_penalty)
    
    return objective + total_penalty

def _objective_function_slsqp(params: np.ndarray) -> float:
    """
    Objective function for local optimization (SLSQP), purely -sum_radii.
    Constraints are handled separately by NonlinearConstraint.
    """
    _, _, _, _, radii, _ = _unpack_params(params)
    return -np.sum(radii)

# Constraint functions for SLSQP (all formulated as g(X) >= 0)
# (Adapted to new _unpack_params and MIN_RADIUS)

# 1. Circle containment constraints
def _circle_containment_constraints(params: np.ndarray) -> np.ndarray:
    rect_width, rect_height, x, y, r, _ = _unpack_params(params)
    
    constraints = np.concatenate([
        x - r,                  # x_i >= r_i
        rect_width - x - r,     # rect_width - x_i >= r_i
        y - r,                  # y_i >= r_i
        rect_height - y - r     # rect_height - y_i >= r_i
    ])
    return constraints

# 2. Non-overlapping circles constraints
# (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
def _non_overlap_constraints(params: np.ndarray) -> np.ndarray:
    _, _, x, y, r, _ = _unpack_params(params)
    
    if N_CIRCLES > 1:
        centers = np.column_stack((x, y))
        dist_sq_matrix = squareform(pdist(centers, metric='sqeuclidean'))
        
        radii_sum_matrix = r[:, None] + r[None, :]
        min_dist_sq_matrix = radii_sum_matrix**2

        upper_tri_indices = np.triu_indices(N_CIRCLES, k=1)
        
        # We need dist_sq_matrix - min_dist_sq_matrix >= 0
        constraint_values = dist_sq_matrix[upper_tri_indices] - min_dist_sq_matrix[upper_tri_indices]
        return constraint_values
    return np.array([]) # No overlap constraints if N_CIRCLES <= 1

# 3. Positive radii constraint
def _positive_radii_constraint(params: np.ndarray) -> np.ndarray:
    _, _, _, _, radii, _ = _unpack_params(params)
    return radii - MIN_RADIUS # r_i >= MIN_RADIUS => r_i - MIN_RADIUS >= 0

# 4. Combined inequality constraints for SLSQP (all g(X) >= 0)
def _all_ineq_constraints_slsqp(params: np.ndarray) -> np.ndarray:
    containment = _circle_containment_constraints(params)
    non_overlap = _non_overlap_constraints(params)
    positive_radii = _positive_radii_constraint(params)
    
    return np.concatenate([containment, non_overlap, positive_radii])

def _is_valid_configuration(params: np.ndarray, tolerance: float = VALIDATION_TOLERANCE) -> bool:
    """
    Checks if a configuration is strictly valid according to all constraints. (From Inspirations 2/3)
    """
    rect_width, rect_height, x, y, r, _ = _unpack_params(params)

    # Check W, H validity based on MIN_RADIUS and perimeter
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
        centers = np.column_stack((x, y))
        dist_sq_matrix = squareform(pdist(centers, metric='sqeuclidean'))
        
        radii_sum_matrix = r[:, None] + r[None, :]
        min_dist_sq_matrix = radii_sum_matrix**2
        
        upper_tri_indices = np.triu_indices(N_CIRCLES, k=1)
        
        if np.any(dist_sq_matrix[upper_tri_indices] < min_dist_sq_matrix[upper_tri_indices] - tolerance):
            return False
                
    return True

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

def circle_packing21(maxiter_de=4000, popsize_de_multiplier=10, maxiter_slsqp=4000, seed=42) -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4
    in order to maximize the sum of their radii. Uses a hybrid optimization strategy
    (Differential Evolution for global search, SLSQP for local refinement).

    Args:
        maxiter_de (int): Maximum number of generations for differential_evolution (global search).
        popsize_de_multiplier (int): Multiplier for the population size for differential_evolution (popsize = multiplier * D).
        maxiter_slsqp (int): Maximum number of iterations for SLSQP (local refinement).
        seed (int): Random seed for reproducibility.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y)
                 coordinates of the i-th circle of radius r.
                 The optimal rectangle dimensions (W, H) are determined internally
                 such that W + H = 2 and are optimized for the packing.
    """
    global _START_TIME # Declare global to modify it for the callback

    _START_TIME = time.time() # Initialize global start time for the callback
    np.random.seed(seed) # Set seed for numpy's random operations

    # Number of dimensions for the optimization problem: 1 (W) + N_CIRCLES * 3 (x,y,r)
    D = 1 + N_CIRCLES * 3 

    # Define bounds for the optimization variables: [W, x1, y1, r1, ..., xN, yN, rN]
    # W: rectangle width. Must be positive, and H=2-W must also be positive.
    width_bounds = (MIN_RADIUS, 2.0 - MIN_RADIUS) # Consistent with Inspirations 2/3
    
    # Max x, y coordinates can go up to (2 - MIN_RADIUS) if the rectangle is very wide/tall.
    x_y_max_bound = 2.0 - MIN_RADIUS
    # Max radius for a single circle in a 1x1 square (W=1, H=1) is 0.5.
    r_max_bound = 0.5 # A generous upper bound for individual radii
    
    circle_bounds = []
    for _ in range(N_CIRCLES):
        circle_bounds.append((0.0, x_y_max_bound)) # x_i
        circle_bounds.append((0.0, x_y_max_bound)) # y_i
        circle_bounds.append((MIN_RADIUS, r_max_bound)) # r_i (must be positive)

    # Combine all bounds: [width_bounds] + [x_i, y_i, r_i bounds for all circles]
    bounds = [width_bounds] + circle_bounds
    
    # Generate an initial population with a heuristic for better starting points (from Inspirations 2/3)
    actual_de_popsize = popsize_de_multiplier * D
    initial_population = np.zeros((actual_de_popsize, D))

    for i in range(actual_de_popsize):
        # Initialize rect_width: sample around 1.0 (square)
        current_width = np.random.uniform(0.8, 1.2) 
        current_width = np.clip(current_width, bounds[0][0], bounds[0][1])
        initial_population[i, 0] = current_width # W is the first parameter

        current_height = 2.0 - current_width

        # Dynamically determine grid dimensions for N_CIRCLES based on aspect ratio
        num_cols = int(np.ceil(np.sqrt(N_CIRCLES * current_width / current_height)))
        num_rows = int(np.ceil(N_CIRCLES / num_cols))
        
        # Calculate cell dimensions. Add small buffer to avoid division by zero.
        safe_width = max(current_width, MIN_RADIUS * 10)
        safe_height = max(current_height, MIN_RADIUS * 10)
        cell_width = safe_width / num_cols
        cell_height = safe_height / num_rows
        
        # Target average radius for initial placement (adjusted from /3.0 to /2.5 from Inspirations 2/3)
        avg_initial_r = np.clip(min(cell_width, cell_height) / 2.5, MIN_RADIUS, r_max_bound)

        # Initialize circles (x, y, r) using a jittered grid
        for j in range(N_CIRCLES):
            idx_x = 1 + j * 3 # +1 because W is at index 0
            idx_y = 1 + j * 3 + 1
            idx_r = 1 + j * 3 + 2

            row = j // num_cols
            col = j % num_cols
            
            base_x = col * cell_width + cell_width / 2.0
            base_y = row * cell_height + cell_height / 2.0

            # Jitter the position within the cell
            jitter_x = np.random.uniform(-cell_width / 4.0, cell_width / 4.0)
            jitter_y = np.random.uniform(-cell_height / 4.0, cell_height / 4.0)

            initial_r = np.random.uniform(MIN_RADIUS, avg_initial_r) 
            
            prop_x = base_x + jitter_x
            prop_y = base_y + jitter_y
            
            initial_r = np.clip(initial_r, MIN_RADIUS, r_max_bound)

            # Define allowed x, y ranges based on current rectangle dimensions and initial_r
            x_min_allowed = initial_r
            x_max_allowed = current_width - initial_r
            y_min_allowed = initial_r
            y_max_allowed = current_height - initial_r

            # Handle cases where initial_r might be too large for the current rectangle dimension
            # Ensure x_min_allowed <= x_max_allowed and y_min_allowed <= y_max_allowed
            if x_min_allowed > x_max_allowed:
                x_center = current_width / 2.0
                x_min_allowed = max(0.0, x_center - MIN_RADIUS) 
                x_max_allowed = min(current_width, x_center + MIN_RADIUS)
            
            if y_min_allowed > y_max_allowed:
                y_center = current_height / 2.0
                y_min_allowed = max(0.0, y_center - MIN_RADIUS)
                y_max_allowed = min(current_height, y_center + MIN_RADIUS)

            prop_x = np.clip(prop_x, x_min_allowed, x_max_allowed)
            prop_y = np.clip(prop_y, y_min_allowed, y_max_allowed)
            
            # Assign to initial population, clipping to overall optimization bounds for safety
            initial_population[i, idx_x] = np.clip(prop_x, bounds[idx_x][0], bounds[idx_x][1])
            initial_population[i, idx_y] = np.clip(prop_y, bounds[idx_y][0], bounds[idx_y][1])
            initial_population[i, idx_r] = initial_r

    # --- Stage 1: Global Search with Differential Evolution ---
    print("--- Starting Stage 1: Global Search (Differential Evolution) ---")

    # Differential Evolution with penalty-based objective (from Inspirations 1/2/3)
    de_result = differential_evolution(
        func=_objective_function_with_penalties,
        bounds=bounds,
        strategy='randtobest1exp', # A robust strategy from Inspiration 2/3
        maxiter=maxiter_de,  # Tuned maxiter for global search
        popsize=popsize_de_multiplier,  # Multiplier for population size
        tol=1e-3,            # Looser tolerance for global search
        seed=seed,
        disp=False,          # Set to True for verbose output
        workers=-1,          # Use all available CPU cores
        polish=True,         # Apply local optimization to DE's best result (from Inspirations 2/3)
        init=initial_population, # Provide the heuristically generated initial population
        callback=_callback_function # Add the time-based callback for early stopping (from Inspiration 1)
    )

    if not de_result.success:
        print(f"Warning: Global search (DE) may not have converged. Message: {de_result.message}")
        print(f"DE best penalized objective: {de_result.fun:.6f}")

    # --- Stage 2: Local Refinement with SLSQP ---
    print("\n--- Starting Stage 2: Local Refinement (SLSQP) ---")
    x0 = de_result.x # Use best result from global search as starting point

    # Define constraints for SLSQP (scipy.optimize.minimize)
    slsqp_constraints = NonlinearConstraint(
        _all_ineq_constraints_slsqp,
        lb=0, # All constraints must be >= 0
        ub=np.inf # No upper bound
    )

    # Run local optimization for fine-tuning
    local_result = minimize(
        fun=_objective_function_slsqp, # Pure -sum_radii objective for SLSQP
        x0=x0,
        method='SLSQP',
        bounds=bounds, # Use the same bounds as DE
        constraints=[slsqp_constraints],
        options={'disp': False, 'maxiter': maxiter_slsqp, 'ftol': 1e-12} # Tight ftol for precision
    )

    # --- Final Result Processing and Validation (from Inspirations 2/3) ---
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
            
    W_opt, H_opt, x_opt, y_opt, r_opt, circles_data_opt = _unpack_params(final_params)
    
    # Explicitly ensure all radii are at least MIN_RADIUS in the output array for robustness
    circles_data_opt[:, 2] = np.maximum(r_opt, MIN_RADIUS)
    
    # One final check on the adjusted circles_array before returning
    final_params_adjusted = np.concatenate(([W_opt], circles_data_opt.flatten()))
    if not _is_valid_configuration(final_params_adjusted, tolerance=VALIDATION_TOLERANCE):
        print("WARNING: Final adjusted configuration is still not strictly valid. Returning it with caution.")

    eval_time = time.time() - _START_TIME # Use the global start time for consistent timing

    # Print final sum of radii and rectangle dimensions
    final_sum_radii = np.sum(circles_data_opt[:, 2])
    print(f"\nOptimization finished.")
    print(f"Optimal sum of radii: {final_sum_radii:.6f}")
    print(f"Optimal rectangle dimensions: W={W_opt:.6f}, H={H_opt:.6f} (W+H={W_opt+H_opt:.4f})")
    print(f"Optimization Time: {eval_time:.4f} seconds")

    return circles_data_opt

# EVOLVE-BLOCK-END

if __name__ == '__main__':
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
