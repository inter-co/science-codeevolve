# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Constants for the problem
N_CIRCLES = 32
RANDOM_SEED = 42 # For reproducibility and deterministic results

def objective_function(x_opt_flat: np.ndarray, N: int) -> float:
    """
    Objective function for scipy.optimize.minimize/differential_evolution.
    Maximizes the sum of radii, so it returns the negative sum of radii.
    """
    # Reshape the flattened array into (N, 3) for (x, y, r) tuples
    circles = x_opt_flat.reshape((N, 3))
    radii = circles[:, 2]
    
    # Ensure radii are non-negative. This is a safeguard; bounds should ideally handle it.
    # Optimization algorithms can sometimes push values slightly out of bounds due to numerical precision.
    non_negative_radii = np.maximum(0, radii)
    
    return -np.sum(non_negative_radii)

def get_constraints(N: int) -> list:
    """
    Generates a list of constraint dictionaries for scipy.optimize.minimize.
    Includes all necessary containment and non-overlap constraints.
    """
    constraints = []

    # 1. Containment constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    # These are formulated as inequality constraints: fun(x) >= 0
    for i in range(N):
        # Extract indices for the i-th circle's x, y, and r
        xi_idx, yi_idx, ri_idx = 3 * i, 3 * i + 1, 3 * i + 2

        # Constraint 1: x_i - r_i >= 0 (left boundary)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx_x=xi_idx, idx_r=ri_idx: x[idx_x] - x[idx_r]})
        # Constraint 2: 1 - x_i - r_i >= 0 (right boundary)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx_x=xi_idx, idx_r=ri_idx: 1 - x[idx_x] - x[idx_r]})
        # Constraint 3: y_i - r_i >= 0 (bottom boundary)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx_y=yi_idx, idx_r=ri_idx: x[idx_y] - x[idx_r]})
        # Constraint 4: 1 - y_i - r_i >= 0 (top boundary)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx_y=yi_idx, idx_r=ri_idx: 1 - x[idx_y] - x[idx_r]})

    # 2. Non-overlap constraints: (xi - xj)² + (yi - yj)² - (ri + rj)² >= 0
    # This ensures the distance between centers is at least the sum of their radii.
    for i in range(N):
        for j in range(i + 1, N): # Iterate over unique pairs (i, j) where i < j
            # Extract indices for circle i
            xi_idx, yi_idx, ri_idx = 3 * i, 3 * i + 1, 3 * i + 2
            # Extract indices for circle j
            xj_idx, yj_idx, rj_idx = 3 * j, 3 * j + 1, 3 * j + 2

            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i_x=xi_idx, i_y=yi_idx, i_r=ri_idx, j_x=xj_idx, j_y=yj_idx, j_r=rj_idx:
                    (x[i_x] - x[j_x])**2 + (x[i_y] - x[j_y])**2 - (x[i_r] + x[j_r])**2
            })
    return constraints

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square, maximizing the sum of their radii.
    This function employs a hybrid optimization strategy:
    1. Global exploration using scipy.optimize.differential_evolution.
    2. Local refinement using scipy.optimize.minimize with the SLSQP method.

    Returns:
        circles: A NumPy array of shape (32, 3), where each row (x, y, r) represents
                 the center coordinates (x, y) and radius (r) of a circle.
    """
    n = N_CIRCLES

    # Define bounds for each variable (x, y, r) for all N circles.
    # x and y coordinates must be within [0, 1].
    # Radius r must be within [0, 0.5] (as diameter cannot exceed the square's side length).
    # For DE, we start with smaller radii to encourage good initial arrangements.
    # For SLSQP, we allow full range for radii.
    
    # Bounds for Differential Evolution (DE) - smaller initial radii range
    de_bounds = [(0, 1), (0, 1), (0, 0.05)] * n # Radii are initially restricted to a small range
    # Bounds for SLSQP - full allowed radii range
    slsqp_bounds = [(0, 1), (0, 1), (0, 0.5)] * n

    # Generate all inequality constraints (containment within square and non-overlap between circles).
    constraints = get_constraints(n)

    print(f"Starting optimization for {n} circles within a unit square.")
    print(f"Total decision variables: {3 * n} (x, y, r for each circle)")
    print(f"Total inequality constraints generated: {len(constraints)}")

    # --- Phase 1: Global Search using Differential Evolution ---
    # Differential Evolution (DE) is a robust global optimization algorithm suitable for multimodal landscapes.
    # It explores the search space to find promising regions, reducing the chance of getting stuck in poor local optima.
    # DE generates its own initial population based on `bounds` and does not require an `x0`.
    print("\n--- Phase 1: Initiating Global Search with scipy.optimize.differential_evolution ---")
    start_time_global = time.time()
    res_global = differential_evolution(
        func=objective_function,
        bounds=de_bounds, # Use restricted bounds for DE to guide initial arrangements
        args=(n,), # Pass N_CIRCLES to the objective function
        strategy='best1bin', # A commonly effective DE strategy
        maxiter=2500, # Increased maxiter for more thorough global exploration
        popsize=15, # Kept popsize as default (15 * D individuals for D=3*N)
        tol=0.001, # Tightened tolerance for DE convergence to find a better global optimum
        mutation=(0.5, 1.0), # Mutation factor range.
        recombination=0.7, # Crossover probability.
        seed=RANDOM_SEED, # Fixed seed for reproducibility.
        disp=True, # Display optimization progress.
        workers=-1 # Use all available CPU cores for parallelization.
    )
    end_time_global = time.time()
    print(f"Differential Evolution completed in {end_time_global - start_time_global:.2f} seconds.")
    print(f"Best sum of radii found by DE: {-res_global.fun:.6f}")
    
    if not res_global.success:
        print(f"Warning: Differential Evolution did not converge successfully: {res_global.message}")
        # Even if not fully converged, `res_global.x` still represents the best solution found.

    # --- Phase 2: Local Refinement using SLSQP ---
    # SLSQP (Sequential Least Squares Programming) is a gradient-based local optimization algorithm
    # well-suited for problems with bounds and a large number of inequality constraints.
    # It starts from the best solution found by DE to precisely fine-tune positions and radii.
    print("\n--- Phase 2: Initiating Local Refinement with scipy.optimize.minimize (SLSQP) ---")
    start_time_local = time.time()
    res_local = minimize(
        fun=objective_function,
        x0=res_global.x, # Start local search from the best solution found by DE.
        args=(n,), # Pass N_CIRCLES to the objective function.
        method='SLSQP',
        bounds=slsqp_bounds, # Use full bounds for SLSQP to allow radii to grow
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 20000, 'disp': True} # Tightened ftol and increased maxiter for more precision.
    )
    end_time_local = time.time()
    print(f"SLSQP completed in {end_time_local - start_time_local:.2f} seconds.")
    print(f"Final sum of radii after SLSQP refinement: {-res_local.fun:.6f}")

    if not res_local.success:
        print(f"Warning: Local optimization (SLSQP) did not converge successfully: {res_local.message}")
        # The result might still be very good, but a warning indicates potential for further improvement.

    # Reshape the final flattened array of variables into the desired (N, 3) format.
    final_circles_flat = res_local.x
    circles = final_circles_flat.reshape((n, 3))

    # Post-processing: Ensure radii are strictly non-negative.
    # This is a final safety check against minor numerical issues.
    circles[:, 2] = np.maximum(0, circles[:, 2])

    return circles

# EVOLVE-BLOCK-END
