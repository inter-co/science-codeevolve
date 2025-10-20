# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from numba import njit # Import njit from numba

# Constants
NUM_CIRCLES = 32
MIN_RADIUS = 1e-6 # Avoid zero radius, but allow very small circles

# Objective function: maximize sum of radii -> minimize negative sum of radii
def objective_function(params: np.ndarray) -> float:
    """
    params: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = params[2::3] # Every 3rd element starting from index 2 is a radius
    return -np.sum(radii)

# Numba-optimized function for constraint evaluation
@njit(cache=True) # Cache compilation for repeated calls to avoid recompilation overhead
def _evaluate_constraints_numba(params: np.ndarray) -> np.ndarray:
    n = NUM_CIRCLES
    
    # Calculate total number of constraints
    num_containment_constraints = 4 * n
    num_overlap_constraints = n * (n - 1) // 2
    total_constraints_count = num_containment_constraints + num_overlap_constraints
    
    constraints_array = np.empty(total_constraints_count, dtype=np.float64)
    
    # Unpack params into x, y, r arrays for easier access.
    # This avoids repeated indexing within loops and improves numba performance.
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    r = np.empty(n, dtype=np.float64)

    for i in range(n):
        x[i] = params[i * 3]
        y[i] = params[i * 3 + 1]
        r[i] = params[i * 3 + 2]

    constraint_idx = 0

    # 1. Containment constraints (x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0)
    for i in range(n):
        constraints_array[constraint_idx] = x[i] - r[i]
        constraint_idx += 1
        constraints_array[constraint_idx] = 1 - x[i] - r[i]
        constraint_idx += 1
        constraints_array[constraint_idx] = y[i] - r[i]
        constraint_idx += 1
        constraints_array[constraint_idx] = 1 - y[i] - r[i]
        constraint_idx += 1

    # 2. Non-overlap constraints ((xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0)
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            constraints_array[constraint_idx] = dist_sq - min_dist_sq
            constraint_idx += 1
            
    return constraints_array

# Constraint function wrapper for scipy.optimize
def total_constraints_func(params: np.ndarray) -> np.ndarray:
    """
    Wrapper function to pass to scipy.optimize.NonlinearConstraint.
    It calls the numba-optimized constraint evaluation function.
    """
    return _evaluate_constraints_numba(params)

# Initial Guess Generation for staggered grid
def generate_initial_guess_staggered_grid(n: int, rows: int, cols: int, initial_r_factor: float = 0.95) -> np.ndarray:
    """
    Generates an initial guess for n circles in a staggered grid pattern.
    This pattern mimics hexagonal packing by offsetting every other row's x-coordinates.
    """
    if rows * cols < n:
        # This case should ideally be avoided by choosing appropriate rows/cols,
        # but the logic handles filling remaining circles if needed.
        pass 

    # Calculate an appropriate initial radius based on grid dimensions.
    # The smallest dimension determines the max possible radius for a 'grid cell'.
    # We use a factor to ensure circles are slightly smaller than tightly packed
    # to give the optimizer room to expand without immediate high violations.
    effective_r_base = 0.5 / max(rows, cols) 
    initial_r = initial_r_factor * effective_r_base
    
    # Ensure initial_r is within valid bounds (positive and not exceeding 0.5 for a single circle)
    initial_r = max(MIN_RADIUS, min(initial_r, 0.5))

    # Calculate step sizes for centers for a uniform grid (before staggering)
    x_grid_span = 1.0 - 2 * initial_r
    y_grid_span = 1.0 - 2 * initial_r

    x_step = x_grid_span / (cols - 1) if cols > 1 else 0.0
    y_step = y_grid_span / (rows - 1) if rows > 1 else 0.0

    initial_circles = []
    current_count = 0
    for i in range(rows):
        y_center = initial_r + i * y_step
        
        # Stagger x-coordinates for hexagonal-like packing: offset every other row
        x_offset = x_step / 2.0 if i % 2 == 1 else 0.0
        
        for j in range(cols):
            if current_count < n:
                x_center = initial_r + j * x_step + x_offset
                
                # Check if the potential circle center would place the circle outside [0,1] bounds
                # This helps prevent initial guesses from having large containment violations.
                if x_center < initial_r or x_center > (1.0 - initial_r):
                    # If staggering pushes a circle out of the primary grid area, skip it
                    continue 
                
                initial_circles.append([x_center, y_center, initial_r])
                current_count += 1
            else:
                break # Reached desired number of circles
        if current_count == n:
            break # Reached desired number of circles
            
    # If we didn't generate enough circles (e.g., due to boundary skips from staggering),
    # fill the remaining slots with small circles at the center.
    while current_count < n:
        initial_circles.append([0.5, 0.5, MIN_RADIUS])
        current_count += 1

    # If we generated more than n circles (e.g., if rows*cols > n), truncate the list
    return np.array(initial_circles[:n]).flatten()

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Uses a scipy.optimize.minimize with SLSQP method, a staggered grid initial guess,
    and Numba-optimized constraint evaluation for performance.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = NUM_CIRCLES

    # 1. Initial Guess Generation
    # For 32 circles, a 6x6 grid offers 36 potential staggered spots.
    # This allows for some flexibility in initial placement and radius.
    initial_guess = generate_initial_guess_staggered_grid(n, rows=6, cols=6, initial_r_factor=0.9)

    # 2. Define Bounds for x, y, r for each circle
    # x, y coordinates must be within [0, 1].
    # Radii must be positive (MIN_RADIUS) and cannot exceed 0.5 (max for a single circle).
    bounds_list = []
    for _ in range(n):
        bounds_list.append((0.0, 1.0)) # x_i
        bounds_list.append((0.0, 1.0)) # y_i
        bounds_list.append((MIN_RADIUS, 0.5)) # r_i
    
    # Create scipy.optimize.Bounds object from the list of (min, max) tuples
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])

    # 3. Define Constraints (Non-linear Inequality Constraints)
    # These include containment within the square and non-overlap between circles.
    # All constraints are formulated as g(x) >= 0.
    num_containment_constraints = 4 * n
    num_overlap_constraints = n * (n - 1) // 2
    total_constraints_count = num_containment_constraints + num_overlap_constraints
    
    # Create a NonlinearConstraint object.
    # `total_constraints_func` returns an array of constraint values.
    # `lb` (lower bound) is an array of zeros, meaning all `g(x) >= 0`.
    # `ub` (upper bound) is an array of infinity, meaning no upper limit on `g(x)`.
    nlc = NonlinearConstraint(total_constraints_func, 
                              np.zeros(total_constraints_count), 
                              np.full(total_constraints_count, np.inf))

    # 4. Optimization using SLSQP (Sequential Least Squares Programming)
    # SLSQP is suitable for constrained, gradient-based optimization.
    # 'maxiter' is increased for potentially better convergence for this complex problem.
    # 'ftol' is the function tolerance.
    # 'disp=False' suppresses optimization output.
    options = {'maxiter': 3000, 'ftol': 1e-8, 'disp': False} 

    res = minimize(objective_function, initial_guess, 
                   method='SLSQP', 
                   bounds=bounds, 
                   constraints=[nlc], 
                   options=options)

    # 5. Process and Return Results
    if not res.success:
        # If optimization fails (e.g., maxiter reached without full convergence),
        # print a warning and return the best found parameters or the initial guess.
        # For evaluation, it's better to return the result of optimization even if not fully successful.
        print(f"Warning: Optimization did not fully converge. Message: {res.message}")
        optimized_params = res.x if res.x is not None else initial_guess
    else:
        optimized_params = res.x

    # Reshape the flattened parameters (x,y,r for all circles) back to (n, 3)
    circles = optimized_params.reshape((n, 3))

    return circles

# EVOLVE-BLOCK-END
