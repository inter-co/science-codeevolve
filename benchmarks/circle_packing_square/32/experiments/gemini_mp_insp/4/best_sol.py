# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
from functools import partial
import time
from joblib import Parallel, delayed # Added for parallelization
from numba import jit # Import numba
import scipy.spatial.distance # Added for vectorized constraints

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
        if r_i < EPSILON_RADIUS - tolerance: # Check if radius is too small
            return False

        # 2. Containment
        # Note: We use EPSILON_RADIUS here to enforce positive radii in bounds,
        # but the actual comparison uses `r_i` which can be slightly larger than EPSILON_RADIUS.
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

# --- Helper for generating diverse initial guesses (adapted from Inspiration 3) ---
def _generate_initial_guess(n_circles: int, seed: int) -> np.ndarray:
    """
    Generates a perturbed grid-based initial guess for N_CIRCLES.
    Ensures initial positions and radii respect bounds.
    """
    rng = np.random.RandomState(seed) # Use a seeded random number generator for reproducibility
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    initial_radius_base = 1.0 / (2 * grid_size)
    
    initial_circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n_circles: break
            
            base_center_x = (2 * i + 1) * initial_radius_base
            base_center_y = (2 * j + 1) * initial_radius_base
            
            # Perturb position slightly within the cell
            perturb_xy_range = (2 * initial_radius_base) * 0.2
            center_x = base_center_x + rng.uniform(-perturb_xy_range, perturb_xy_range)
            center_y = base_center_y + rng.uniform(-perturb_xy_range, perturb_xy_range)

            # Perturb radius slightly
            perturb_r_range = initial_radius_base * 0.25
            radius = initial_radius_base + rng.uniform(-perturb_r_range, perturb_r_range)

            # Clip to ensure the initial guess respects the absolute bounds
            radius = np.clip(radius, EPSILON_RADIUS, 0.5 - EPSILON_RADIUS) # Use global EPSILON_RADIUS
            center_x = np.clip(center_x, radius, 1 - radius)
            center_y = np.clip(center_y, radius, 1 - radius)

            initial_circles[count] = [center_x, center_y, radius]
            count += 1
        if count >= n_circles: break
            
    return initial_circles.flatten()

# --- Main Constructor Function ---
def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    This implementation uses scipy.optimize.basinhopping with an SLSQP solver,
    and parallelizes multiple runs with diverse initial guesses, leveraging Numba for performance.
    It adopts vectorized constraints and Jacobians from Inspiration 3 for efficiency.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES # Use global constant
    
    # 2. Objective Function: Minimize negative sum of radii
    @jit(nopython=True, fastmath=True) # JIT compile objective
    def objective(params):
        # Radii are at indices 2, 5, 8, ...
        radii = params[2::3]
        return -np.sum(radii)

    # Gradient of the objective function
    @jit(nopython=True, fastmath=True) # JIT compile gradient
    def gradient_objective(params):
        grad = np.zeros_like(params)
        # The objective is -sum(radii)
        # Radii are at indices 2, 5, 8, ... (i*3 + 2)
        grad[2::3] = -1.0 # Optimized with direct slicing
        return grad

    # 3. Vectorized Constraint Functions and Jacobian (Adapted from Inspiration 3)
    # These functions are NOT JIT-compiled as they rely on NumPy's vectorized operations
    # and scipy.optimize.minimize often prefers standard Python callables for constraints.
    def _vectorized_constraints(params, n):
        """Computes all constraint values as a single numpy array."""
        circles = params.reshape((n, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        # Containment constraints (4*n constraints)
        c_contain_x_min = x - r
        c_contain_x_max = 1.0 - x - r
        c_contain_y_min = y - r
        c_contain_y_max = 1.0 - y - r

        # Non-overlap constraints (n*(n-1)/2 constraints)
        num_pairs = n * (n - 1) // 2
        c_overlap = np.zeros(num_pairs)
        if n > 1:
            centers = circles[:, :2]
            sq_distances = scipy.spatial.distance.pdist(centers, 'sqeuclidean') # Vectorized squared distance
            
            # Efficiently compute (ri+rj)^2 for all pairs
            radii_sums = np.add.outer(r, r)[np.triu_indices(n, k=1)]
            sq_radii_sums = radii_sums ** 2
            c_overlap = sq_distances - sq_radii_sums

        return np.concatenate([
            c_contain_x_min, c_contain_x_max, 
            c_contain_y_min, c_contain_y_max, 
            c_overlap
        ])

    def _vectorized_jacobian(params, n):
        """Computes the Jacobian of the vectorized constraints."""
        num_params = 3 * n
        num_pairs = n * (n - 1) // 2
        num_constraints = 4 * n + num_pairs
        jac = np.zeros((num_constraints, num_params))
        
        circles = params.reshape((n, 3))
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

        row_offset = 0
        # Part 1: Jacobian of containment constraints
        for i in range(n): jac[row_offset + i, 3*i] = 1.0; jac[row_offset + i, 3*i+2] = -1.0 # d(x_i - r_i)
        row_offset += n
        for i in range(n): jac[row_offset + i, 3*i] = -1.0; jac[row_offset + i, 3*i+2] = -1.0 # d(1 - x_i - r_i)
        row_offset += n
        for i in range(n): jac[row_offset + i, 3*i+1] = 1.0; jac[row_offset + i, 3*i+2] = -1.0 # d(y_i - r_i)
        row_offset += n
        for i in range(n): jac[row_offset + i, 3*i+1] = -1.0; jac[row_offset + i, 3*i+2] = -1.0 # d(1 - y_i - r_i)
        row_offset += n

        # Part 2: Jacobian of non-overlap constraints
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                sr = r[i] + r[j]
                
                jac[row_offset + pair_idx, 3*i] = 2 * dx      # d/dx_i
                jac[row_offset + pair_idx, 3*i+1] = 2 * dy    # d/dy_i
                jac[row_offset + pair_idx, 3*i+2] = -2 * sr   # d/dr_i
                
                jac[row_offset + pair_idx, 3*j] = -2 * dx     # d/dx_j
                jac[row_offset + pair_idx, 3*j+1] = -2 * dy   # d/dy_j
                jac[row_offset + pair_idx, 3*j+2] = -2 * sr   # d/dr_j
                pair_idx += 1
        return jac

    # Consolidated constraints for scipy.optimize.minimize
    constraints = [{
        'type': 'ineq', 
        'fun': partial(_vectorized_constraints, n=n), 
        'jac': partial(_vectorized_jacobian, n=n)
    }]

    # 4. Bounds for x, y, r (adjusted for clarity and consistency)
    bounds = []
    for _ in range(n):
        bounds.append((EPSILON_RADIUS, 1.0 - EPSILON_RADIUS)) # x_i
        bounds.append((EPSILON_RADIUS, 1.0 - EPSILON_RADIUS)) # y_i
        bounds.append((EPSILON_RADIUS, 0.5 - EPSILON_RADIUS)) # r_i (max radius 0.5 for a single circle)

    # Helper function for parallel execution of a single basinhopping run
    def _run_basinhopping_single_seed(seed, n, objective, gradient_objective, constraints, bounds, minimizer_options, niter, T, stepsize_bh):
        
        # Generate a unique perturbed initial guess for this specific run
        x0_perturbed = _generate_initial_guess(n, seed)

        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": constraints,
            "options": minimizer_options,
            "jac": gradient_objective
        }

        bh_result = basinhopping(
            func=objective,
            x0=x0_perturbed, # Start each basinhopping run from its own perturbed initial guess
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            T=T,
            stepsize=stepsize_bh, # Use basinhopping's default step with the specified stepsize
            seed=seed, # Use the specific seed for this run's basinhopping internal randomness
            disp=False # Turn off individual basinhopping disp to avoid clutter
        )

        if bh_result.x is not None:
            current_sum_radii = -objective(bh_result.x)
        else:
            current_sum_radii = -np.inf # Indicate failure to find parameters

        return current_sum_radii, bh_result.x

    # 5. Perform Optimization using basinhopping with parallel runs
    minimizer_options = {'ftol': 1e-8, 'disp': False, 'maxiter': 2000}
    
    niter_per_run = 250 # Number of basinhopping iterations per parallel run
    T = 1.0
    stepsize_bh = 0.02 # Step size for basinhopping (for its internal default step function)

    num_parallel_runs = 4 # Retained 4 parallel runs as in previous best context

    # Generate distinct seeds for each parallel run to ensure diverse exploration
    seeds = [RANDOM_SEED_BASE + i for i in range(num_parallel_runs)]

    print(f"Starting {num_parallel_runs} parallel basinhopping runs, each with {niter_per_run} iterations...")

    # Use joblib to run basinhopping in parallel
    results = Parallel(n_jobs=-1)( # n_jobs=-1 uses all available CPU cores
        delayed(_run_basinhopping_single_seed)(
            seed, n, objective, gradient_objective, constraints, bounds,
            minimizer_options, niter_per_run, T, stepsize_bh
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
        print("Warning: All parallel optimization runs failed or returned no valid parameters. Attempting single minimize run on a stable initial grid.")
        # Fallback to a single minimize run with a deterministic initial guess
        fallback_initial_params = _generate_initial_guess(n, RANDOM_SEED_BASE)
        result = minimize(
            objective,
            fallback_initial_params,
            method='SLSQP',
            jac=gradient_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False}
        )
        if result.success:
            optimized_params = result.x
            print(f"Fallback minimize run successful with sum_radii: {-objective(optimized_params):.6f}")
        else:
            print("Warning: Fallback minimize run also failed. Returning an unoptimized initial grid packing.")
            optimized_params = fallback_initial_params
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
