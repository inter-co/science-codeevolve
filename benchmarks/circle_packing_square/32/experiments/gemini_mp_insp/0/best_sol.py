# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping # Added basinhopping
from functools import partial
from joblib import Parallel, delayed # Added for parallelization
from scipy.spatial.distance import pdist # For vectorized distance calculation

# For determinism
np.random.seed(42)

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    This implementation uses scipy.optimize.basinhopping for global optimization,
    with a local SLSQP solver that uses analytical gradients and vectorized constraints,
    executed in parallel using joblib. This approach is inspired by the best practices
    from the provided inspiration programs to achieve high sum_radii efficiently.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # 1. Initial Guess Generation (Perturbed Grid, inspired by Inspiration 3)
    # This function generates a different starting point for each parallel run.
    def _generate_initial_guess(n_circles, seed):
        rng = np.random.RandomState(seed) # Use a seeded random number generator for reproducibility
        grid_size = 6
        initial_radius = 1.0 / (2 * grid_size)
        
        initial_circles = np.zeros((n_circles, 3))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles: break
                
                base_center_x = (2 * i + 1) * initial_radius
                base_center_y = (2 * j + 1) * initial_radius
                
                # Perturb position slightly; cell size is 2*initial_radius
                perturb_xy_range = (2 * initial_radius) * 0.2
                center_x = base_center_x + rng.uniform(-perturb_xy_range, perturb_xy_range)
                center_y = base_center_y + rng.uniform(-perturb_xy_range, perturb_xy_range)

                # Perturb radius slightly
                perturb_r_range = initial_radius * 0.25
                radius = initial_radius + rng.uniform(-perturb_r_range, perturb_r_range)

                # Clip to ensure the initial guess respects the absolute bounds
                epsilon_local = 1e-7 # Use a local epsilon to avoid conflict with the one for bounds
                radius = np.clip(radius, epsilon_local, 0.5 - epsilon_local)
                center_x = np.clip(center_x, radius, 1 - radius)
                center_y = np.clip(center_y, radius, 1 - radius)

                initial_circles[count] = [center_x, center_y, radius]
                count += 1
            if count >= n_circles: break
                
        return initial_circles.flatten()

    # 2. Objective Function and Gradient
    def objective(params):
        return -np.sum(params[2::3])

    def gradient_objective(params, n_circles): # Modified to accept n_circles
        grad = np.zeros_like(params)
        grad[2::3] = -1.0
        return grad

    # 3. Vectorized Constraint Functions and Jacobian (Inspired by Inspiration Program 1)
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
            sq_distances = pdist(centers, 'sqeuclidean') # Vectorized squared distance
            
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

    # 4. Bounds for x, y, r
    epsilon = 1e-7 
    bounds = []
    for _ in range(n):
        bounds.append((epsilon, 1.0 - epsilon)) # x_i
        bounds.append((epsilon, 1.0 - epsilon)) # y_i
        bounds.append((epsilon, 0.5 - epsilon)) # r_i (max radius 0.5 for a single circle)

    # 5. Perform Global Optimization using basinhopping with parallel runs (Inspired by Inspiration Program 1)
    minimizer_options = {'ftol': 1e-8, 'disp': False, 'maxiter': 2000}
    
    # Configure basinhopping parameters
    niter_per_run = 250 # Number of basinhopping iterations per parallel run
    T = 1.0 # Temperature for Metropolis criterion
    stepsize = 0.02 # Step size for random displacement

    num_parallel_runs = 4 # Number of parallel basinhopping instances

    # Helper function for parallel execution of a single basinhopping run
    def _run_basinhopping_single_seed(seed, n, objective, gradient_objective, constraints, bounds, minimizer_options, niter, T, stepsize):
        # Generate a unique perturbed initial guess for this run
        initial_params_flat = _generate_initial_guess(n_circles=n, seed=seed)

        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": constraints,
            "options": minimizer_options,
            "jac": partial(gradient_objective, n_circles=n) # Pass n_circles to gradient
        }

        bh_result = basinhopping(
            func=objective,
            x0=initial_params_flat,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            T=T,
            stepsize=stepsize,
            seed=seed,
            disp=False
        )

        current_sum_radii = -objective(bh_result.x) if bh_result.x is not None else -np.inf
        return current_sum_radii, bh_result.x

    # Generate distinct seeds for each parallel run
    base_seed = 42
    seeds = [base_seed + i for i in range(num_parallel_runs)]

    # Use joblib to run basinhopping in parallel, with progress logging
    print(f"Starting {num_parallel_runs} parallel basinhopping runs, each with {niter_per_run} iterations...")
    results = Parallel(n_jobs=-1)( # n_jobs=-1 uses all available CPU cores
        delayed(_run_basinhopping_single_seed)(
            seed, n, objective, gradient_objective, constraints, bounds,
            minimizer_options, niter_per_run, T, stepsize
        ) for seed in seeds
    )
    print("Parallel basinhopping runs finished.")

    best_sum_radii = -np.inf
    best_optimized_params = None

    # Aggregate results from all parallel runs
    for current_sum_radii, current_params in results:
        if current_params is not None and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_optimized_params = current_params

    # 6. Post-processing and Fallback
    if best_optimized_params is None:
        # If all parallel basinhopping runs failed, fall back to a single minimize run on a stable initial grid.
        print("Warning: All parallel basinhopping runs failed. Attempting single minimize run on a stable initial grid.")
        # Generate a deterministic initial guess for the fallback using a fixed seed.
        fallback_initial_params = _generate_initial_guess(n_circles=n, seed=42)

        result = minimize(
            objective,
            fallback_initial_params,
            method='SLSQP',
            jac=partial(gradient_objective, n_circles=n),
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False}
        )
        if result.success:
            best_optimized_params = result.x
        else:
            print("Warning: Single minimize run also failed. Returning the fallback initial grid packing.")
            best_optimized_params = fallback_initial_params
    else:
        # Add a print statement for the best result found, inspired by Inspiration 1
        print(f"Overall best sum_radii found = {best_sum_radii:.6f}")

    final_circles = best_optimized_params.reshape((n, 3))
    # Ensure radii are not negative due to numerical precision
    final_circles[:, 2] = np.maximum(final_circles[:, 2], epsilon) 
    
    return final_circles

# EVOLVE-BLOCK-END
