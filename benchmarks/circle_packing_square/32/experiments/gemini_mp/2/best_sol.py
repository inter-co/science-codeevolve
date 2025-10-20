# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import numba # Import numba for JIT compilation

@numba.jit(nopython=True, fastmath=True) # Apply Numba JIT for performance
def _objective_function_de(params, n_circles=32, penalty_weight=1e6):
    """
    Objective function for differential_evolution.
    Minimizes -(sum of radii) + penalty for constraint violations.
    """
    circles = params.reshape((n_circles, 3))
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    # 1. Sum of radii (to maximize, so we minimize its negative)
    sum_radii = np.sum(rs)
    objective = -sum_radii

    # 2. Containment constraints
    # r <= x <= 1-r  =>  x - r >= 0  AND  1 - r - x >= 0
    # r <= y <= 1-r  =>  y - r >= 0  AND  1 - r - y >= 0
    containment_violations = np.sum(np.maximum(0, rs - xs)) + \
                             np.sum(np.maximum(0, xs + rs - 1)) + \
                             np.sum(np.maximum(0, rs - ys)) + \
                             np.sum(np.maximum(0, ys + rs - 1))

    # 3. Non-overlap constraints
    overlap_violations = 0.0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist_req_sq = (rs[i] + rs[j])**2
            overlap_violations += np.maximum(0.0, min_dist_req_sq - dist_sq) # Penalize if dist_sq < min_dist_req_sq

    # Total penalty
    penalty = penalty_weight * (containment_violations + overlap_violations)

    return objective + penalty

@numba.jit(nopython=True, fastmath=True) # Apply Numba JIT for performance
def _refinement_objective_slsqp(params, n_circles):
    """
    Objective function for local refinement (SLSQP).
    Minimizes -(sum of radii) directly, constraints are handled explicitly.
    """
    circles = params.reshape((n_circles, 3))
    rs = circles[:, 2]
    return -np.sum(rs)

@numba.jit(nopython=True, fastmath=True) # Apply Numba JIT for performance
def _constraint_function_slsqp(params, n_circles):
    """
    Constraint function for local refinement (SLSQP).
    Returns an array of values, all of which must be >= 0 for validity.
    """
    circles = params.reshape((n_circles, 3))
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    num_constraints = 4 * n_circles + num_overlap_constraints
    constraints_arr = np.empty(num_constraints, dtype=params.dtype)
    
    idx = 0
    # Containment: r <= x <= 1-r  =>  x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    for i in range(n_circles):
        constraints_arr[idx] = xs[i] - rs[i]
        idx += 1
        constraints_arr[idx] = 1 - xs[i] - rs[i]
        idx += 1
        constraints_arr[idx] = ys[i] - rs[i]
        idx += 1
        constraints_arr[idx] = 1 - ys[i] - rs[i]
        idx += 1

    # Non-overlap: dist_ij >= r_i + r_j  =>  dist_ij^2 - (r_i + r_j)^2 >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            sum_radii_sq = (rs[i] + rs[j])**2
            constraints_arr[idx] = dist_sq - sum_radii_sq
            idx += 1
            
    return constraints_arr

def _validate_circles(circles, n_circles=32, tolerance=1e-7):
    """
    Validates a set of circle configurations for containment and non-overlap.
    """
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    if np.any(rs < -tolerance):
        print("WARNING: Negative radius detected!")
        return 0.0, False, False

    containment_valid = True
    for i in range(n_circles):
        if not (rs[i] - tolerance <= xs[i] <= 1 - rs[i] + tolerance and \
                rs[i] - tolerance <= ys[i] <= 1 - rs[i] + tolerance):
            containment_valid = False
            break

    overlap_valid = True
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < rs[i] + rs[j] - tolerance:
                overlap_valid = False
                break
        if not overlap_valid:
            break

    sum_radii = np.sum(rs)
    return sum_radii, containment_valid, overlap_valid

def _generate_initial_population(n_circles, pop_size, dim, seed):
    """
    Generates a diverse and semi-feasible initial population for DE.
    This is a key improvement to guide the optimizer effectively.
    """
    rng = np.random.default_rng(seed)
    population = np.zeros((pop_size, dim))

    # Strategy 1: A structured, grid-based guess for the first individual.
    # This provides a very strong, feasible starting point.
    grid_dim = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_dim
    radius = spacing / 2.0 * 0.98 # Start with radii guaranteed not to overlap
    
    params_grid = np.zeros(dim)
    count = 0
    for i in range(grid_dim):
        for j in range(grid_dim):
            if count < n_circles:
                idx = count * 3
                params_grid[idx] = i * spacing + spacing / 2.0  # x
                params_grid[idx+1] = j * spacing + spacing / 2.0 # y
                params_grid[idx+2] = radius
                count += 1
    population[0] = params_grid

    # Strategy 2: Random, sparse placements for the rest of the population.
    # This ensures diversity and explores other regions of the search space.
    for i in range(1, pop_size):
        params_rand = np.zeros(dim)
        params_rand[0::3] = rng.uniform(0.1, 0.9, n_circles)  # x
        params_rand[1::3] = rng.uniform(0.1, 0.9, n_circles)  # y
        params_rand[2::3] = rng.uniform(0.01, 0.05, n_circles) # small initial radii
        population[i] = params_rand
        
    return population

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    """
    n = 32
    dim = n * 3
    seed = 42

    # KEY IMPROVEMENT 1: Tighter radius bound significantly helps the optimizer.
    # Theoretical max radius for 32 circles is ~0.1, so 0.15 is a safe upper bound.
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.15)] * n

    # Differential Evolution parameters
    de_popsize_multiplier = 10
    de_maxiter = 300
    de_popsize = de_popsize_multiplier * dim

    print(f"Starting Differential Evolution with popsize={de_popsize}, maxiter={de_maxiter}...")
    
    # KEY IMPROVEMENT 2: Generate a smart initial population to guide the global search.
    print("Generating custom initial population (grid-based + random)...")
    initial_population = _generate_initial_population(n, de_popsize, dim, seed)

    # Run Differential Evolution with the custom initial population
    de_result = differential_evolution(
        _objective_function_de,
        bounds,
        args=(n,),
        strategy='best1bin',
        maxiter=de_maxiter,
        popsize=de_popsize_multiplier,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        disp=False,
        workers=-1,
        init=initial_population # Use our custom population
    )

    print("Differential Evolution finished.")
    optimal_params_de = de_result.x
    
    # Local Refinement using SLSQP
    print("Starting local refinement with SLSQP...")
    constraints_slsqp = [{'type': 'ineq', 'fun': _constraint_function_slsqp, 'args': (n,)}]

    refinement_result = minimize(
        _refinement_objective_slsqp,
        optimal_params_de,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_slsqp,
        options={'maxiter': 1000, 'ftol': 1e-7, 'disp': False}
    )
    print("Local refinement finished.")

    optimal_params = refinement_result.x
    optimal_circles = optimal_params.reshape((n, 3))

    # Validate the final configuration
    sum_radii, containment_valid, overlap_valid = _validate_circles(optimal_circles, n)

    print(f"Optimization complete. Final sum of radii: {sum_radii:.6f}")
    print(f"Containment valid: {containment_valid}")
    print(f"Non-overlap valid: {overlap_valid}")
    
    if not (containment_valid and overlap_valid):
        print("WARNING: Final solution has minor constraint violations.")
        # Attempt to fix minor violations by slightly reducing all radii
        for _ in range(5):
            sum_radii, containment_valid, overlap_valid = _validate_circles(optimal_circles, n)
            if containment_valid and overlap_valid:
                break
            optimal_circles[:, 2] *= 0.9999
        sum_radii = np.sum(optimal_circles[:, 2])
        print(f"After fix attempt: Sum radii = {sum_radii:.6f}, Valid = {containment_valid and overlap_valid}")

    # Ensure radii are non-negative as a final safeguard
    optimal_circles[:, 2] = np.maximum(0, optimal_circles[:, 2])

    return optimal_circles

# EVOLVE-BLOCK-END
