# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import time
from numba import njit # Import numba for performance

# Helper function to extract (x, y, r) from the flattened vector
# Numba-optimized version for internal use within jitted functions
@njit(cache=True)
def unpack_circles_numba(x_vec, n_circles):
    x = np.empty(n_circles, dtype=x_vec.dtype)
    y = np.empty(n_circles, dtype=x_vec.dtype)
    r = np.empty(n_circles, dtype=x_vec.dtype)
    for i in range(n_circles):
        x[i] = x_vec[i*3]
        y[i] = x_vec[i*3 + 1]
        r[i] = x_vec[i*3 + 2]
    return x, y, r

# Objective function: minimize -sum(radii)
# This function is called by scipy.optimize, so it doesn't need to be jitted itself,
# but it calls numba-optimized helpers if needed.
def objective(x_vec, n_circles):
    # Using the standard numpy reshape as this function is not the bottleneck
    r = x_vec.reshape((n_circles, 3))[:, 2]
    return -np.sum(r)

# Constraint function for all nonlinear constraints
# Numba-optimized for performance, especially for the overlap checks
@njit(cache=True)
def constraints_fun_numba(x_vec, n_circles):
    x, y, r = unpack_circles_numba(x_vec, n_circles)
    
    # Max constraints: 4*n (boundary) + n*(n-1)/2 (overlap)
    # For n=32: 4*32 + 32*31/2 = 128 + 496 = 624
    # Pre-allocate a large enough array.
    constraints_array = np.empty(624, dtype=x_vec.dtype)
    idx = 0

    # 1. Boundary containment constraints (4 per circle: x-r, y-r, 1-x-r, 1-y-r >= 0)
    for i in range(n_circles):
        constraints_array[idx] = x[i] - r[i]
        idx += 1
        constraints_array[idx] = y[i] - r[i]
        idx += 1
        constraints_array[idx] = 1.0 - x[i] - r[i]
        idx += 1
        constraints_array[idx] = 1.0 - y[i] - r[i]
        idx += 1

    # 2. Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r[i] + r[j])**2
            constraints_array[idx] = dist_sq - min_dist_sq
            idx += 1
            
    return constraints_array[:idx] # Return only the filled part of the array

# Wrapper for scipy.optimize to call the numba-compiled function
def constraints_fun(x_vec, n_circles):
    return constraints_fun_numba(x_vec, n_circles)

# Initial guess strategy: hexagonal-like packing
def get_initial_guess(n_circles, random_seed=42):
    np.random.seed(random_seed) # Ensure determinism
    
    # Estimate average radius based on benchmark to guide initial packing
    # 2.937 / 32 ~ 0.0917. Start slightly smaller to allow expansion.
    initial_r = 0.088 # A reasonable initial radius for hexagonal packing
    
    circles_data = []
    
    # Hexagonal grid parameters
    dx = 2 * initial_r # Horizontal spacing between centers
    dy = np.sqrt(3) * initial_r # Vertical spacing between centers for hexagonal pattern
    x_offset_row = initial_r # Offset for alternating rows

    current_y = initial_r
    circle_count = 0
    row_num = 0

    # Define circle counts per row to approximate 32 circles hexagonally
    # This arrangement aims to fill the square somewhat efficiently.
    # Total: 6+5+6+5+6+4 = 32 circles
    row_counts = [6, 5, 6, 5, 6, 4] 
    
    for count_in_row in row_counts:
        current_x = initial_r
        if row_num % 2 != 0: # Odd rows are shifted horizontally
            current_x += x_offset_row
            
        for _ in range(count_in_row):
            if circle_count >= n_circles:
                break
            circles_data.append([current_x, current_y, initial_r])
            current_x += dx
            circle_count += 1
        
        current_y += dy
        row_num += 1
        if circle_count >= n_circles:
            break

    circles_data = np.array(circles_data)
    
    # Add small random perturbation to break perfect symmetry and help optimizer escape local minima
    # and to ensure radii are not all identical initially.
    circles_data[:, :2] += np.random.uniform(-0.005, 0.005, (n_circles, 2))
    circles_data[:, 2] += np.random.uniform(-0.001, 0.001, n_circles) 

    # Clip values to ensure they are within valid ranges
    # Radii must be positive and not exceed 0.5 (half the square side)
    circles_data[:, 0] = np.clip(circles_data[:, 0], initial_r, 1 - initial_r)
    circles_data[:, 1] = np.clip(circles_data[:, 1], initial_r, 1 - initial_r)
    circles_data[:, 2] = np.clip(circles_data[:, 2], 1e-6, 0.5) 

    return circles_data.flatten()

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Bounds for each variable (x, y, r) for all circles
    # x: [0, 1], y: [0, 1], r: [1e-6, 0.5]
    bounds = Bounds(
        lb=[0.0, 0.0, 1e-6] * n,
        ub=[1.0, 1.0, 0.5] * n
    )

    # Nonlinear constraints
    # All constraints are of the form C(x) >= 0, so lower bound is 0, upper bound is infinity.
    n_boundary_constraints = 4 * n
    n_overlap_constraints = n * (n - 1) // 2
    total_constraints = n_boundary_constraints + n_overlap_constraints
    
    nonlinear_constraints = NonlinearConstraint(
        fun=lambda x_vec: constraints_fun(x_vec, n),
        lb=np.zeros(total_constraints, dtype=np.float64),
        ub=np.full(total_constraints, np.inf, dtype=np.float64),
        # Jacobians can be provided for SLSQP for better performance, but '2-point' approximation is default
        # and often sufficient for first attempts.
        # jac='2-point'
    )

    # Initial guess for the optimizer
    x0 = get_initial_guess(n, random_seed=42) # Use fixed seed for reproducibility

    # Optimization using SLSQP (Sequential Least Squares Programming)
    # It's a local optimizer suitable for constrained problems.
    # Options are tuned for better convergence.
    result = minimize(
        fun=lambda x_vec: objective(x_vec, n),
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraints],
        options={
            'maxiter': 2000, # Increased max iterations for more thorough search
            'ftol': 1e-8,    # Function tolerance for convergence
            'disp': False    # Set to True for verbose output
        }
    )

    if not result.success:
        print(f"Optimization failed: {result.message}. Current sum_radii: {-result.fun:.4f}")
        # If optimization fails, attempt to return the best found configuration
        # after ensuring radii are positive.
        circles = result.x.reshape((n, 3))
        circles[:, 2] = np.maximum(circles[:, 2], 1e-6) # Ensure radii are positive
    else:
        circles = result.x.reshape((n, 3))
    
    # Final check of constraints and radii
    # Small numerical violations can occur, ensure radii are positive and
    # constraints are met within a small tolerance.
    final_violations = constraints_fun(circles.flatten(), n)
    if np.any(final_violations < -1e-5): # Check for significant violations
        print(f"Warning: Constraints violated after optimization. Max violation: {np.min(final_violations):.2e}")
        # A simple fix for small violations: if a circle slightly overlaps, its radius might be too large.
        # Or if it's slightly outside bounds. No automatic fix here, just a warning.
        # Ensure all radii are at least 1e-6
        circles[:,2] = np.maximum(circles[:,2], 1e-6) 
    
    # Ensure circles count is exactly 32
    if circles.shape[0] != n or circles.shape[1] != 3:
        raise ValueError(f"Expected {n} circles of (x,y,r), got shape {circles.shape}")

    return circles

# EVOLVE-BLOCK-END
