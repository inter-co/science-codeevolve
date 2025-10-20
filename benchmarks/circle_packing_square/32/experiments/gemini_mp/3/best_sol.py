# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from scipy.spatial.distance import pdist, squareform

def _objective_function_circle_packing(params, n_circles, penalty_coeff):
    """
    Objective function for the GLOBAL search (differential_evolution).
    It's the negative sum of radii plus a penalty for constraint violations.
    """
    circles_config = params.reshape(n_circles, 3)
    x, y, r = circles_config[:, 0], circles_config[:, 1], circles_config[:, 2]
    objective_val = -np.sum(r)
    penalty = 0.0
    
    # Containment constraints
    penalty += np.sum(np.maximum(0, r - x))
    penalty += np.sum(np.maximum(0, x + r - 1))
    penalty += np.sum(np.maximum(0, r - y))
    penalty += np.sum(np.maximum(0, y + r - 1))
    
    # Non-overlap constraints
    if n_circles > 1:
        centers = circles_config[:, :2]
        dist_centers = squareform(pdist(centers))
        radii_sum_pairs = r[:, np.newaxis] + r[np.newaxis, :]
        i, j = np.triu_indices(n_circles, k=1)
        overlap_violations = np.maximum(0, radii_sum_pairs[i, j] - dist_centers[i, j])
        penalty += np.sum(overlap_violations)

    return objective_val + penalty_coeff * penalty

def _local_objective(params):
    """Objective function for the LOCAL search (SLSQP), no penalty term."""
    n_circles = len(params) // 3
    radii = params.reshape(n_circles, 3)[:, 2]
    return -np.sum(radii)

def _all_constraints(params):
    """Function defining all constraints for the LOCAL search (SLSQP)."""
    n_circles = len(params) // 3
    circles = params.reshape(n_circles, 3)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # Containment constraints (g(x) >= 0) -> 4*N constraints
    containment = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

    # Non-overlap constraints (g(x) >= 0) -> N*(N-1)/2 constraints
    if n_circles > 1:
        centers = circles[:, :2]
        # Use squared distances to avoid sqrt, which is faster and monotonic
        dist_sq = squareform(pdist(centers, 'sqeuclidean'))
        i, j = np.triu_indices(n_circles, k=1)
        radii_sum_pairs = r[i] + r[j]
        # Constraint: dist_sq[i,j] - (r[i]+r[j])^2 >= 0
        overlap = dist_sq[i, j] - (radii_sum_pairs)**2
    else:
        overlap = np.array([])
        
    return np.concatenate([containment, overlap])

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses a two-stage optimization process:
    1. A long, aggressive global search with differential_evolution to find a
       promising region.
    2. A precise local search with SLSQP using hard non-linear constraints to
       refine the solution to an optimum.
    """
    N_CIRCLES = 32
    POPSIZE = 150 # Further increased population size to enhance global exploration
    
    bounds_per_circle = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] 
    all_bounds = bounds_per_circle * N_CIRCLES

    # --- Create a structured initial population for the global search ---
    grid_size = 6
    x_coords = np.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size)
    y_coords = np.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    initial_centers = grid_points[:N_CIRCLES, :]
    initial_radii = np.full((N_CIRCLES, 1), 0.05)
    grid_guess = np.hstack([initial_centers, initial_radii]).flatten()
    
    n_params = N_CIRCLES * 3
    rng = np.random.default_rng(seed=42)
    init_pop = rng.random((POPSIZE, n_params))
    lows = np.array([b[0] for b in all_bounds])
    highs = np.array([b[1] for b in all_bounds])
    init_pop = lows + init_pop * (highs - lows)
    init_pop[0, :] = grid_guess

    penalty_coeff = 1e8

    # --- Stage 1: Aggressive Global Search with Differential Evolution ---
    # Maxiter and Popsize are further tuned to maximize utilization of the 180-second time budget.
    result_de = differential_evolution(
        func=_objective_function_circle_packing,
        bounds=all_bounds,
        args=(N_CIRCLES, penalty_coeff),
        strategy='best1bin',
        maxiter=9000,         # Increased maxiter to allow more generations for global exploration
        popsize=POPSIZE,      # Increased population size for greater diversity
        tol=1e-6,             # Maintained tight tolerance
        init=init_pop,
        disp=False,
        workers=-1,
        updating='deferred',
        seed=42
    )
    
    initial_guess_for_local = result_de.x

    # --- Stage 2: Local Refinement with SLSQP and Hard Constraints ---
    nonlinear_constraint = NonlinearConstraint(_all_constraints, 0, np.inf)

    result_local = minimize(
        fun=_local_objective,
        x0=initial_guess_for_local,
        method='SLSQP',
        bounds=all_bounds,
        constraints=[nonlinear_constraint],
        options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False} # Increased maxiter for even more thorough local refinement
    )

    best_params = result_local.x
    circles = best_params.reshape(N_CIRCLES, 3)
    circles[:, 2] = np.maximum(circles[:, 2], 1e-9)

    return circles

# EVOLVE-BLOCK-END
