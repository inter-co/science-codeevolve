# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
from functools import partial
import time
from joblib import Parallel, delayed # Added for parallelization
from numba import jit # Import numba

# --- Global Constants (inspired by Inspiration 2 & 3) ---
N_CIRCLES = 32
EPSILON_RADIUS = 1e-7 # Minimum allowed radius for a circle to be considered 'real'
RANDOM_SEED_BASE = 42 # Base seed for reproducibility


# --- JIT-compiled Helper for Validation (inspired by Inspiration 3) ---
@jit(nopython=True, fastmath=True)
def is_valid_packing(circles: np.ndarray, tolerance: float = 1e-7) -> bool:
    """
    Checks if a given configuration of circles is valid (non-overlapping and contained).
    This function is JIT compiled for performance.

    Args:
        circles: np.array of shape (N,3), where each row is (x,y,r).
        tolerance: A small floating-point tolerance for comparisons.

    Returns:
        True if the packing is valid, False otherwise.
    """
    n = circles.shape[0]

    for i in range(n):
        x_i, y_i, r_i = circles[i]

        # 1. Positive Radii
        if r_i <= EPSILON_RADIUS - tolerance: # Use <= as in Inspiration 1, slightly more robust
            return False

        # 2. Containment
        # The `tolerance` is added/subtracted to allow for floating point inaccuracies
        # near the boundary.
        if not (r_i - tolerance <= x_i <= 1 - r_i + tolerance and \
                r_i - tolerance <= y_i <= 1 - r_i + tolerance):
            return False

    # 3. Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            x_i, y_i, r_i = circles[i]
            x_j, y_j, r_j = circles[j]

            dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
            min_dist_sq = (r_i + r_j)**2

            if dist_sq < min_dist_sq - tolerance: # If actual distance is less than required, it overlaps
                return False
                
    return True

# --- Main Constructor Function ---
def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    This implementation uses scipy.optimize.basinhopping with an SLSQP solver,
    and parallelizes multiple runs. It leverages Numba for performance and
    starts basinhopping runs from the same initial grid for consistency.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES # Use global constant
    
    # 1. Initial Guess Generation (from the original grid packing logic)
    grid_size = int(np.ceil(np.sqrt(n))) # Dynamically determine grid size
    initial_radius = 1.0 / (2 * grid_size)

    initial_circles = np.zeros((n, 3))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            
            center_x = (2 * i + 1) * initial_radius
            center_y = (2 * j + 1) * initial_radius
            
            initial_circles[count] = [center_x, center_y, initial_radius]
            count += 1
        if count >= n:
            break
            
    # Flatten the initial guess for the optimizer
    # params structure: [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1]
    initial_params_flat = initial_circles.flatten()

    # 2. Objective Function: Minimize negative sum of radii
    @jit(nopython=True, fastmath=True) # JIT compile objective
    def objective(params):
        # Radii are at indices 2, 5, 8, ...
        radii = params[2::3]
        return -np.sum(radii)

    # Gradient of the objective function (aligned with Inspiration 1)
    @jit(nopython=True, fastmath=True) # JIT compile gradient
    def gradient_objective(params): # Removed n_circles parameter
        grad = np.zeros_like(params)
        # The objective is -sum(radii)
        # Radii are at indices 2, 5, 8, ... (i*3 + 2)
        grad[2::3] = -1.0
        return grad

    # 3. Constraint Functions
    constraints = []

    # Containment constraints (ri <= xi <= 1-ri and ri <= yi <= 1-ri)
    @jit(nopython=True, fastmath=True)
    def _containment_x_min(params, i):
        return params[i*3] - params[i*3 + 2] # x_i - r_i
    @jit(nopython=True, fastmath=True)
    def _containment_x_max(params, i):
        return 1 - params[i*3] - params[i*3 + 2] # 1 - x_i - r_i
    @jit(nopython=True, fastmath=True)
    def _containment_y_min(params, i):
        return params[i*3 + 1] - params[i*3 + 2] # y_i - r_i
    @jit(nopython=True, fastmath=True)
    def _containment_y_max(params, i):
        return 1 - params[i*3 + 1] - params[i*3 + 2] # 1 - y_i - r_i
    
    # Gradients for containment constraints
    @jit(nopython=True, fastmath=True)
    def jac_containment_x_min(params, i):
        grad = np.zeros_like(params)
        grad[i*3] = 1.0  # d/dx_i
        grad[i*3 + 2] = -1.0 # d/dr_i
        return grad

    @jit(nopython=True, fastmath=True)
    def jac_containment_x_max(params, i):
        grad = np.zeros_like(params)
        grad[i*3] = -1.0 # d/dx_i
        grad[i*3 + 2] = -1.0 # d/dr_i
        return grad

    @jit(nopython=True, fastmath=True)
    def jac_containment_y_min(params, i):
        grad = np.zeros_like(params)
        grad[i*3 + 1] = 1.0 # d/dy_i
        grad[i*3 + 2] = -1.0 # d/dr_i
        return grad

    @jit(nopython=True, fastmath=True)
    def jac_containment_y_max(params, i):
        grad = np.zeros_like(params)
        grad[i*3 + 1] = -1.0 # d/dy_i
        grad[i*3 + 2] = -1.0 # d/dr_i
        return grad

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': partial(_containment_x_min, i=i), 'jac': partial(jac_containment_x_min, i=i)})
        constraints.append({'type': 'ineq', 'fun': partial(_containment_x_max, i=i), 'jac': partial(jac_containment_x_max, i=i)})
        constraints.append({'type': 'ineq', 'fun': partial(_containment_y_min, i=i), 'jac': partial(jac_containment_y_min, i=i)})
        constraints.append({'type': 'ineq', 'fun': partial(_containment_y_max, i=i), 'jac': partial(jac_containment_y_max, i=i)})

    # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    @jit(nopython=True, fastmath=True)
    def _non_overlap(params, i, j):
        x_i, y_i, r_i = params[i*3 : i*3 + 3]
        x_j, y_j, r_j = params[j*3 : j*3 + 3]
        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
        min_dist_sq = (r_i + r_j)**2
        return dist_sq - min_dist_sq

    # Gradient for non-overlap constraint
    @jit(nopython=True, fastmath=True)
    def jac_non_overlap(params, i, j):
        grad = np.zeros_like(params)
        x_i, y_i, r_i = params[i*3 : i*3 + 3]
        x_j, y_j, r_j = params[j*3 : j*3 + 3]

        grad[i*3] = 2 * (x_i - x_j)             # d/dx_i
        grad[i*3 + 1] = 2 * (y_i - y_j)         # d/dy_i
        grad[i*3 + 2] = -2 * (r_i + r_j)        # d/dr_i

        grad[j*3] = -2 * (x_i - x_j)            # d/dx_j
        grad[j*3 + 1] = -2 * (y_i - y_j)        # d/dy_j
        grad[j*3 + 2] = -2 * (r_i + r_j)        # d/dr_j
        return grad

    for i in range(n):
        for j in range(i + 1, n): # Only unique pairs
            constraints.append({'type': 'ineq', 'fun': partial(_non_overlap, i=i, j=j), 'jac': partial(jac_non_overlap, i=i, j=j)})

    # 4. Bounds for x, y, r
    # x, y: [0, 1] (containment constraints will enforce stricter bounds based on r)
    # r: (EPSILON_RADIUS, 0.5 - EPSILON_RADIUS) to ensure positivity and prevent trivial solutions
    bounds = []
    for _ in range(n):
        bounds.append((EPSILON_RADIUS, 1.0 - EPSILON_RADIUS)) # x_i
        bounds.append((EPSILON_RADIUS, 1.0 - EPSILON_RADIUS)) # y_i
        bounds.append((EPSILON_RADIUS, 0.5 - EPSILON_RADIUS)) # r_i (max radius is 0.5 if centered)

    # Helper function for parallel execution of a single basinhopping run
    def _run_basinhopping_single_seed(seed, n, objective, gradient_objective, constraints, bounds, initial_params_flat, minimizer_options, niter, T, stepsize):
        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": constraints,
            "options": minimizer_options,
            "jac": gradient_objective # Corrected to pass the function directly, as it no longer needs n_circles
        }

        # Use the passed seed for basinhopping
        bh_result = basinhopping(
            func=objective,
            x0=initial_params_flat, # Start each basinhopping run from the SAME grid packing (reverted to previous best strategy)
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            T=T,
            stepsize=stepsize, # Use basinhopping's default step-taking with this stepsize
            seed=seed, # Use the specific seed for this run
            disp=False # Turn off individual basinhopping disp to avoid clutter
        )

        if bh_result.x is not None:
            current_sum_radii = -objective(bh_result.x)
        else:
            current_sum_radii = -np.inf # Indicate failure to find parameters

        return current_sum_radii, bh_result.x

    # 5. Perform Optimization using basinhopping with parallel runs
    minimizer_options = {'ftol': 1e-8, 'disp': False, 'maxiter': 2000}
    
    niter_per_run = 300 # Slightly increased number of basinhopping iterations per parallel run
    T = 1.0
    stepsize = 0.02 # Basinhopping's default step-taking uses this parameter

    num_parallel_runs = 8 # Increased number of parallel runs for more global exploration

    # Generate distinct seeds for each parallel run to ensure diverse exploration
    seeds = [RANDOM_SEED_BASE + i for i in range(num_parallel_runs)]

    print(f"Starting {num_parallel_runs} parallel basinhopping runs, each with {niter_per_run} iterations...")

    # Use joblib to run basinhopping in parallel
    results = Parallel(n_jobs=-1)( # n_jobs=-1 uses all available CPU cores
        delayed(_run_basinhopping_single_seed)(
            seed, n, objective, gradient_objective, constraints, bounds,
            initial_params_flat, minimizer_options, niter_per_run, T, stepsize
        ) for seed in seeds
    )
    print("Parallel basinhopping runs finished.")

    best_sum_radii = -np.inf
    best_circles_params = None

    # Aggregate results from all parallel runs
    for current_sum_radii, current_params in results:
        if current_params is not None and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles_params = current_params

    # 6. Post-processing and Error Handling
    if best_circles_params is None:
        print("Warning: All parallel optimization runs failed or returned no valid parameters. Returning the unperturbed initial guess.")
        optimized_params = initial_params_flat
    else:
        optimized_params = best_circles_params
        print(f"Overall best sum_radii found = {best_sum_radii:.6f}")

    circles = optimized_params.reshape((n, 3))

    # Ensure radii are not negative or excessively small due to numerical precision
    circles[:, 2] = np.maximum(circles[:, 2], EPSILON_RADIUS) # Use global constant

    # Final validation and reporting (inspired by Inspiration 3)
    print("\n--- Final Results Validation ---")
    if is_valid_packing(circles, tolerance=1e-6):
        print("Final packing is valid!")
    else:
        print("WARNING: Final packing is NOT strictly valid according to `is_valid_packing`.")
        print("This may be due to floating point tolerances or the optimizer finishing near a constraint boundary.")

    final_sum_radii = np.sum(circles[:, 2])
    print(f"Final sum of radii (after post-processing): {final_sum_radii:.6f}")
    benchmark = 2.937
    print(f"Benchmark to beat: {benchmark}")
    print(f"Benchmark ratio: {final_sum_radii / benchmark:.6f}")

    return circles

# EVOLVE-BLOCK-END
