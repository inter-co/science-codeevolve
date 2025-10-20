# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit

# Numba-optimized function for distance and overlap check
@njit(cache=True)
def _calculate_violations(params, n_circles):
    """
    Calculates squared distances between circle centers and squared sum of radii.
    Returns an array of overlap violations: (r_i + r_j)^2 - ((x_i - x_j)^2 + (y_i - y_j)^2).
    A value > 0 indicates overlap.
    """
    circles = params.reshape(n_circles, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    violations = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            radii_sum_sq = (r[i] + r[j])**2
            violations.append(radii_sum_sq - dist_sq)
    return np.array(violations)

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # 1. Objective function: Minimize negative sum of radii
    def neg_sum_radii(params):
        radii = params[2::3]
        return -np.sum(radii)

    # 2. Initial Guess: Hexagonal-like arrangement
    np.random.seed(42) # for determinism
    
    # Base radius for initial hexagonal packing
    # This value is a rough estimate to get a dense, non-overlapping starting configuration
    # For N=32, a common density value suggests r around 0.08-0.09.
    r_initial_base = 0.085 
    
    # Define rows for hexagonal packing. Total 32 circles.
    # Example: 6, 5, 6, 5, 6, 4 = 32 circles
    # This aims for a somewhat square-like aspect ratio (6 rows, max 6 circles)
    circles_per_row = [6, 5, 6, 5, 6, 4] 
    
    circles_list = []
    current_y = r_initial_base
    y_step = r_initial_base * np.sqrt(3) # Vertical distance between centers in hex packing
    x_step = 2 * r_initial_base          # Horizontal distance between centers

    for row_idx, num_circles_in_row in enumerate(circles_per_row):
        # Alternate x-offset for hexagonal pattern
        x_offset = r_initial_base if row_idx % 2 == 0 else r_initial_base + r_initial_base # Start at r or r+r
        
        for col_idx in range(num_circles_in_row):
            x = x_offset + col_idx * x_step
            circles_list.append([x, current_y, r_initial_base])
        current_y += y_step

    initial_circles_arr = np.array(circles_list)

    # Scale the initial packing to fit within the unit square [0,1]x[0,1]
    # Find the extent of the current packing
    min_x = np.min(initial_circles_arr[:, 0] - initial_circles_arr[:, 2])
    max_x = np.max(initial_circles_arr[:, 0] + initial_circles_arr[:, 2])
    min_y = np.min(initial_circles_arr[:, 1] - initial_circles_arr[:, 2])
    max_y = np.max(initial_circles_arr[:, 1] + initial_circles_arr[:, 2])

    # Calculate scaling factor to fit within [0,1]
    scale_factor = 1.0 / max(max_x - min_x, max_y - min_y)
    
    # Apply scaling and recenter
    initial_circles_arr[:, :2] = (initial_circles_arr[:, :2] - [min_x, min_y]) * scale_factor
    initial_circles_arr[:, 2] *= scale_factor # Scale radii as well
    
    # A small perturbation to break perfect symmetry and help the optimizer explore
    perturbation_scale = 0.005
    initial_circles_arr[:, :2] += np.random.uniform(-perturbation_scale, perturbation_scale, (n, 2))
    initial_circles_arr[:, 2] += np.random.uniform(-perturbation_scale/2, perturbation_scale/2, n)
    
    # Ensure radii are positive and positions are within square after perturbation
    initial_circles_arr[:, 2] = np.maximum(initial_circles_arr[:, 2], 1e-6)
    initial_circles_arr[:, 0] = np.clip(initial_circles_arr[:, 0], initial_circles_arr[:, 2], 1 - initial_circles_arr[:, 2])
    initial_circles_arr[:, 1] = np.clip(initial_circles_arr[:, 1], initial_circles_arr[:, 2], 1 - initial_circles_arr[:, 2])

    initial_params = initial_circles_arr.flatten()

    # 3. Bounds for (x, y, r)
    # x and y must be between 0 and 1. r must be positive. Max r is 0.5.
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((1e-6, 0.5)) # r_i (radius must be positive, max 0.5 for a single circle)

    # 4. Constraints
    constraints = []

    # a. Boundary containment constraints (ri <= xi <= 1-ri and ri <= yi <= 1-ri)
    # Formulated as: xi - ri >= 0, 1 - xi - ri >= 0, yi - ri >= 0, 1 - yi - ri >= 0
    # Using separate lambda functions for each constraint.
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda params, i=i: params[i*3] - params[i*3 + 2]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda params, i=i: 1 - params[i*3] - params[i*3 + 2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda params, i=i: params[i*3 + 1] - params[i*3 + 2]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda params, i=i: 1 - params[i*3 + 1] - params[i*3 + 2]})

    # b. Non-overlap constraints (dist_sq - (r_i + r_j)^2 >= 0)
    # The _calculate_violations function returns (r_i + r_j)^2 - dist_sq.
    # So we need to ensure this is <= 0 for all pairs, meaning -violations >= 0.
    constraints.append({'type': 'ineq', 'fun': lambda params: -_calculate_violations(params, n)})

    # 5. Optimization
    # Using SLSQP, which is suitable for small-to-medium sized constrained problems.
    # Increased maxiter and ftol for better convergence.
    result = minimize(
        neg_sum_radii,
        initial_params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-8, 'disp': False} # Increased maxiter and ftol
    )

    if not result.success:
        print(f"Optimization warning: {result.message}")
        # Consider re-running with different initial conditions or method if result is poor.
    
    circles = result.x.reshape(n, 3)

    # Post-processing: ensure radii are not tiny due to numerical issues
    circles[:, 2] = np.maximum(circles[:, 2], 1e-7) # slightly smaller minimum radius

    return circles

# EVOLVE-BLOCK-END
