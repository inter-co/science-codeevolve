# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit
import time

# Use a fixed random seed for reproducibility
np.random.seed(42)

# Numba-jitted function for fast overlap checking
@njit(cache=True)
def _check_overlaps_numba(params: np.ndarray, n: int) -> np.ndarray:
    """
    Checks for overlaps between circles and returns squared distances minus squared sum of radii.
    A value < 0 means overlap (violation).
    Constraint is formulated as: dist_sq - (ri+rj)^2 >= 0
    """
    violations = []
    # Reshape params for easier access (x, y, r for each circle)
    circles_data = params.reshape(n, 3)
    
    for i in range(n):
        xi, yi, ri = circles_data[i, 0], circles_data[i, 1], circles_data[i, 2]
        for j in range(i + 1, n): # Check each unique pair
            xj, yj, rj = circles_data[j, 0], circles_data[j, 1], circles_data[j, 2]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (ri + rj)**2
            
            # Constraint: dist_sq - min_dist_sq >= 0
            violations.append(dist_sq - min_dist_sq)
            
    return np.array(violations)

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # 1. Initial Placement Strategy (Hexagonal-like grid)
    # Arrange 26 circles in a [5, 6, 5, 6, 4] pattern across 5 rows to approximate hexagonal packing.
    row_counts = [5, 6, 5, 6, 4] # Total 26 circles
    num_rows = len(row_counts)
    
    # Calculate initial radius based on fitting the widest row with offset
    # The widest span for centers occurs when an odd row (6 circles) is offset by `r_val`.
    # Overall horizontal span including radii: `3*r_val + (k_max-1)*2*r_val <= 1`
    # `3*r_val + (6-1)*2*r_val <= 1` => `3*r_val + 10*r_val <= 1` => `13*r_val <= 1`
    # `r_val <= 1/13 = 0.07692...`
    r_val = 0.076 # Slightly less than 1/13 to ensure fit
    
    dx = 2 * r_val # Horizontal spacing between centers in a row
    dy = np.sqrt(3) * r_val # Vertical spacing between row centers for hexagonal packing

    # Calculate overall pattern dimensions for centering
    # Max x-coordinate reached by a circle's right edge, assuming left-alignment and odd row offset
    x_pattern_width = 3 * r_val + (max(row_counts) - 1) * dx
    # Max y-coordinate reached by a circle's top edge, assuming bottom-alignment
    y_pattern_height = 2 * r_val + (num_rows - 1) * dy
    
    # Centering offsets
    x_center_offset = (1.0 - x_pattern_width) / 2.0
    y_center_offset = (1.0 - y_pattern_height) / 2.0
    
    circles_initial = np.zeros((n, 3))
    current_circle_idx = 0
    
    # Current y-position for placing the row (center of circles)
    current_y_pos = y_center_offset + r_val 
    
    for i, count in enumerate(row_counts):
        # Calculate x_start for current row, applying hexagonal offset
        # Even rows start closer to the left edge of the pattern
        # Odd rows are shifted right by `r_val`
        x_start_row = x_center_offset + r_val if i % 2 == 0 else x_center_offset + r_val + r_val
        
        x_current = x_start_row
        for j in range(count):
            if current_circle_idx < n:
                circles_initial[current_circle_idx, 0] = x_current
                circles_initial[current_circle_idx, 1] = current_y_pos
                # Introduce slight radius variation based on row density to guide the optimizer
                if count < 6: # Circles in less dense rows (5 or 4) can start slightly larger
                    circles_initial[current_circle_idx, 2] = r_val * 1.02
                else: # Circles in the most dense rows (6) start slightly smaller
                    circles_initial[current_circle_idx, 2] = r_val * 0.98
                current_circle_idx += 1
                x_current += dx
            else:
                break # Should not happen if row_counts sum to n
        current_y_pos += dy

    # Add small random perturbation to initial parameters to aid optimization
    perturb_scale_pos = 0.005 * r_val # e.g., 0.5% of r_val
    perturb_scale_r = 0.001 * r_val # e.g., 0.1% of r_val
    
    perturbations = np.zeros_like(circles_initial)
    perturbations[:, 0] = np.random.uniform(-perturb_scale_pos, perturb_scale_pos, n) # x
    perturbations[:, 1] = np.random.uniform(-perturb_scale_pos, perturb_scale_pos, n) # y
    perturbations[:, 2] = np.random.uniform(-perturb_scale_r, perturb_scale_r, n) # r
    
    initial_params = (circles_initial + perturbations).flatten()

    # 2. Objective Function: Minimize negative sum of radii
    def objective(params: np.ndarray) -> float:
        radii = params[2::3] # Radii are every 3rd element starting from index 2
        return -np.sum(radii)

    # 3. Constraints
    constraints = []

    # Bounds for x, y, r for each circle
    # x and y must be within [0, 1]
    # r must be positive (e.g., >= 1e-6) and not exceed 0.5 (max for a single circle)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((1e-6, 0.5)) # r_i (radius must be positive and within reasonable max)

    # Define constraint function for SLSQP
    # This function should return an array of values that must be >= 0
    def _constraint_func(params: np.ndarray) -> np.ndarray:
        violations = []
        circles_data = params.reshape(n, 3)

        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        for i in range(n):
            xi, yi, ri = circles_data[i, 0], circles_data[i, 1], circles_data[i, 2]
            violations.append(xi - ri)      # x_i - r_i >= 0
            violations.append(1 - xi - ri)  # 1 - x_i - r_i >= 0
            violations.append(yi - ri)      # y_i - r_i >= 0
            violations.append(1 - yi - ri)  # 1 - y_i - r_i >= 0
        
        # Overlap constraints (using numba-jitted function for speed)
        overlap_violations = _check_overlaps_numba(params, n)
        
        # Combine all constraints into a single array
        return np.concatenate((np.array(violations), overlap_violations))

    # Constraint dictionary for scipy.optimize.minimize
    # type='ineq' means func(x) >= 0
    constraint_dict = {'type': 'ineq', 'fun': _constraint_func}
    
    # 4. Optimization
    # Using SLSQP for constrained local optimization.
    # It's sensitive to initial guess and can get stuck in local optima.
    # A good initial guess (like the hexagonal pattern) is crucial.
    
    # --- Optimization using Iterated Local Search ---
    # This strategy runs the optimizer once, then repeatedly perturbs the best-found
    # solution and re-optimizes. This helps escape shallow local optima and find a
    # more globally competitive solution without the high cost of a full global optimizer.

    # 4.1. Initial Optimization Run from Hexagonal Guess
    # Use more iterations and a tighter tolerance for a better initial convergence.
    options = {'maxiter': 15000, 'ftol': 1e-9, 'disp': False} 
    
    result = minimize(
        objective,
        initial_params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraint_dict,
        options=options
    )

    if not result.success:
        print(f"Initial optimization failed: {result.message}")
        # If the first run fails, we still use its result as a starting point for refinement.
    
    best_params = result.x
    best_score = result.fun if result.success else objective(result.x)

    # 4.2. Iterative Refinement via Perturbation and Re-optimization
    # Increase restarts and make perturbation more aggressive and adaptive.
    num_restarts = 8 # More restarts for better exploration of the solution space
    for i in range(num_restarts):
        current_circles = best_params.reshape(n, 3)
        r_mean = np.mean(current_circles[:, 2])
        
        # Define perturbation scale based on the current average radius. This adaptive
        # perturbation is larger than the initial one to effectively "kick" the
        # solution out of a local minimum.
        # We also introduce a factor that slightly increases perturbation strength
        # in later restarts to encourage broader exploration if stuck.
        perturb_factor = 1.0 + 0.5 * (i / num_restarts)
        perturb_scale_pos = perturb_factor * 0.025 * r_mean 
        perturb_scale_r = perturb_factor * 0.015 * r_mean
        
        perturbations = np.zeros_like(current_circles)
        perturbations[:, 0] = np.random.uniform(-perturb_scale_pos, perturb_scale_pos, n)
        perturbations[:, 1] = np.random.uniform(-perturb_scale_pos, perturb_scale_pos, n)
        # Bias radius perturbations towards shrinking to create more "space" for the
        # optimizer to work with, while still allowing for some growth.
        perturbations[:, 2] = np.random.uniform(-2 * perturb_scale_r, perturb_scale_r, n)
        
        re_initial_params = (current_circles + perturbations).flatten()

        # Clip the perturbed parameters to ensure they remain within the defined bounds.
        # This prevents starting the next optimization from a grossly invalid state.
        for j in range(len(re_initial_params)):
            lower, upper = bounds[j]
            re_initial_params[j] = np.clip(re_initial_params[j], lower, upper)

        # Re-run optimization from the perturbed state. More iterations are needed
        # to properly re-converge after a stronger perturbation.
        restart_options = {'maxiter': 7500, 'ftol': 1e-9, 'disp': False}
        
        restart_result = minimize(
            objective,
            re_initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_dict,
            options=restart_options
        )
        
        # If the new run converged and found a better solution, update our best.
        if restart_result.success and restart_result.fun < best_score:
            best_params = restart_result.x
            best_score = restart_result.fun
    
    optimized_circles = best_params.reshape(n, 3)
    
    # Ensure all radii are explicitly positive, even if very small (due to 1e-6 lower bound, they should be)
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 1e-6)

    return optimized_circles

# EVOLVE-BLOCK-END