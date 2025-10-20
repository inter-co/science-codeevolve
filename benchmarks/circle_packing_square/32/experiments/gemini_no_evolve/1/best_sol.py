# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from numba import njit
import time

# --- Numba JIT compiled functions for performance ---
# These functions are critical for performance as they are called repeatedly
# by the optimizer for constraint evaluation.
@njit(cache=True)
def _eval_non_overlap_constraints_numba(params_xyr):
    """
    Evaluates the non-overlap constraints for all pairs of circles.
    Constraint: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    """
    n = len(params_xyr) // 3
    num_pairs = n * (n - 1) // 2
    constraints = np.zeros(num_pairs)
    k = 0
    for i in range(n):
        for j in range(i + 1, n): # Only check each pair once
            x_i, y_i, r_i = params_xyr[i*3], params_xyr[i*3+1], params_xyr[i*3+2]
            x_j, y_j, r_j = params_xyr[j*3], params_xyr[j*3+1], params_xyr[j*3+2]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2
            constraints[k] = dist_sq - min_dist_sq # Must be >= 0
            k += 1
    return constraints

@njit(cache=True)
def _eval_boundary_constraints_numba(params_xyr):
    """
    Evaluates the boundary containment constraints for all circles.
    Constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    """
    n = len(params_xyr) // 3
    constraints = np.zeros(4 * n) # 4 constraints per circle
    
    for i in range(n):
        x_i, y_i, r_i = params_xyr[i*3], params_xyr[i*3+1], params_xyr[i*3+2]
        
        # x-bounds
        constraints[i*4 + 0] = x_i - r_i
        constraints[i*4 + 1] = 1.0 - x_i - r_i
        # y-bounds
        constraints[i*4 + 2] = y_i - r_i
        constraints[i*4 + 3] = 1.0 - y_i - r_i
    return constraints

# Objective function for scipy.optimize.minimize
def objective(params_xyr):
    """
    Objective function to minimize. Maximizing sum of radii is equivalent
    to minimizing the negative sum of radii.
    """
    radii = params_xyr[2::3] # Extract all radii
    return -np.sum(radii)

# Wrapper functions for NonlinearConstraint
def non_overlap_constraints_func(params_xyr):
    return _eval_non_overlap_constraints_numba(params_xyr)

def boundary_constraints_func(params_xyr):
    return _eval_boundary_constraints_numba(params_xyr)

# Initial guess generation for N=32 using a hexagonal-like pattern
def generate_initial_guess_hexagonal(n_circles, r_initial_guess=0.08):
    """
    Generates an initial guess for circle positions and radii in a hexagonal-like pattern.
    For N=32, it uses a specific row distribution [6, 5, 6, 5, 6, 4].
    """
    if n_circles != 32:
        # Fallback for other N: simple square grid
        side_len = int(np.ceil(np.sqrt(n_circles)))
        r_grid = 0.5 / side_len
        x_centers = np.linspace(r_grid, 1-r_grid, side_len)
        y_centers = np.linspace(r_grid, 1-r_grid, side_len)
        
        circles_xyr = []
        count = 0
        for y in y_centers:
            for x in x_centers:
                if count < n_circles:
                    circles_xyr.extend([x, y, r_grid * 0.95]) # Slightly smaller radius to start
                    count += 1
        return np.array(circles_xyr)
        
    # Specific hexagonal-like pattern for N=32, aiming for better density
    target_row_lengths = [6, 5, 6, 5, 6, 4] # Sums to 32 circles
    
    circles_xyr = []
    
    # Calculate y-coordinates for each row based on hexagonal vertical spacing
    y_coords_rows = []
    current_y = r_initial_guess
    for _ in range(len(target_row_lengths)):
        y_coords_rows.append(current_y)
        current_y += r_initial_guess * np.sqrt(3) # Vertical spacing for hexagonal packing
        
    # Center the entire block of rows vertically within the unit square
    # Total height of the packing block (from bottom of first circle to top of last circle)
    total_height = y_coords_rows[-1] + r_initial_guess - y_coords_rows[0]
    y_shift = (1.0 - total_height) / 2.0
    y_coords_rows = [y + y_shift for y in y_coords_rows]
    
    circle_count = 0
    for row_idx, num_in_row in enumerate(target_row_lengths):
        if circle_count >= n_circles:
            break
        
        y_center = y_coords_rows[row_idx]
        
        # Calculate x-coordinates for circles in the current row
        # Unstaggered rows (even row_idx: 0, 2, 4) span from r_initial_guess to 1-r_initial_guess
        # Staggered rows (odd row_idx: 1, 3, 5) are shifted horizontally by r_initial_guess.
        # To ensure containment, the effective x-range for centers of staggered rows is smaller.
        if row_idx % 2 == 0: # Unstaggered row
            x_coords = np.linspace(r_initial_guess, 1.0 - r_initial_guess, num_in_row)
        else: # Staggered row
            # Shifted by r_initial_guess, so the effective x-range starts at 2*r_initial_guess
            # and ends at 1 - 2*r_initial_guess.
            x_coords = np.linspace(2 * r_initial_guess, 1.0 - 2 * r_initial_guess, num_in_row)
        
        for x_center in x_coords:
            if circle_count < n_circles:
                circles_xyr.extend([x_center, y_center, r_initial_guess])
                circle_count += 1
            else:
                break
                                
    return np.array(circles_xyr)

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a constrained optimization approach with a hexagonal-like initial guess.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Set a fixed random seed for reproducibility in initial guess generation and restarts
    np.random.seed(42)

    # Initial guess generation: a hexagonal-like pattern for N=32
    # An initial radius of 0.08 is chosen as a reasonable starting point
    # that allows the target number of circles to fit without severe initial overlaps.
    initial_params_xyr = generate_initial_guess_hexagonal(n, r_initial_guess=0.08)
    
    # Ensure the initial guess array has the correct size (N*3)
    if len(initial_params_xyr) < n * 3:
        # Pad with zeros if initial guess generator couldn't fill all N circles
        # (shouldn't happen for N=32 with current logic, but good for robustness)
        initial_params_xyr = np.pad(initial_params_xyr, (0, n * 3 - len(initial_params_xyr)), 
                                    'constant', constant_values=0)
    
    # Define bounds for each variable (x, y, r)
    # x_i, y_i must be within [0, 1]. r_i must be positive and <= 0.5.
    # A small lower bound for radius (1e-6) prevents numerical issues with zero radii.
    bounds_list = []
    for _ in range(n):
        bounds_list.append((0.0, 1.0)) # x_i coordinate
        bounds_list.append((0.0, 1.0)) # y_i coordinate
        bounds_list.append((1e-6, 0.5)) # r_i radius (must be positive, max 0.5)
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])

    # Define nonlinear constraints for the optimizer
    # Constraints are defined in the form g(x) >= 0.
    
    # 1. Non-overlap constraints for all pairs of circles
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    non_overlap_nlc = NonlinearConstraint(non_overlap_constraints_func, 0, np.inf)
    
    # 2. Boundary containment constraints for each circle
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    boundary_nlc = NonlinearConstraint(boundary_constraints_func, 0, np.inf)

    # Combine all nonlinear constraints
    constraints = [non_overlap_nlc, boundary_nlc]

    # Optimization using the SLSQP method
    # SLSQP is suitable for constrained optimization problems, including those with nonlinear constraints.
    # It's a local optimizer, so the quality of the initial guess is crucial.
    # 'maxiter' is increased to allow more iterations for convergence.
    # 'ftol' is the function tolerance for termination.
    options = {'maxiter': 2500, 'ftol': 1e-9, 'disp': False} 

    # Perform the initial optimization run
    result = minimize(
        objective,
        initial_params_xyr,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options
    )

    best_sum_radii = -result.fun if result.success else -np.inf
    best_circles_xyr = result.x if result.success else initial_params_xyr
    
    # Multiple restarts heuristic to escape local minima
    # Perturb the best solution found so far and re-optimize.
    # This helps explore the solution space more thoroughly.
    num_restarts = 5 # Number of restart attempts
    for _ in range(num_restarts):
        # Apply a small random perturbation to the current best solution
        # This creates a new starting point for the optimizer.
        perturbed_params = best_circles_xyr + np.random.uniform(-0.02, 0.02, size=best_circles_xyr.shape)
        
        # Ensure radii remain positive and within sensible bounds after perturbation
        perturbed_params[2::3] = np.clip(perturbed_params[2::3], 1e-6, 0.5)
        # Clip coordinates to stay within [0,1]
        perturbed_params[0::3] = np.clip(perturbed_params[0::3], 0.0, 1.0) # x
        perturbed_params[1::3] = np.clip(perturbed_params[1::3], 0.0, 1.0) # y
        
        # Re-run optimization from the perturbed starting point
        restart_result = minimize(
            objective,
            perturbed_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        # Update best solution if a better one is found
        if restart_result.success and -restart_result.fun > best_sum_radii:
            best_sum_radii = -restart_result.fun
            best_circles_xyr = restart_result.x
            
    # Reshape the optimal parameters into the required (n, 3) format
    optimal_circles = best_circles_xyr.reshape((n, 3))
    
    # Final check: ensure all radii are non-negative
    optimal_circles[:, 2] = np.maximum(optimal_circles[:, 2], 0.0)

    return optimal_circles

# EVOLVE-BLOCK-END
