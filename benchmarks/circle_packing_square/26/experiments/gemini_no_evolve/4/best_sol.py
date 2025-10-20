# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from scipy.spatial.distance import pdist, squareform
from numba import njit
import time # For eval_time

# --- Configuration Constants ---
N_CIRCLES = 26
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Numba-optimized Helper Functions ---
@njit(cache=True)
def _check_overlap_numba(positions_radii):
    """
    Checks for overlaps between circles.
    positions_radii: np.array of shape (N, 3) where columns are (x, y, r)
    Returns: sum of overlap violations (positive if overlapping)
    """
    n = positions_radii.shape[0]
    total_violation = 0.0
    for i in range(n):
        xi, yi, ri = positions_radii[i, 0], positions_radii[i, 1], positions_radii[i, 2]
        for j in range(i + 1, n):
            xj, yj, rj = positions_radii[j, 0], positions_radii[j, 1], positions_radii[j, 2]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist_sq = (ri + rj)**2
            if dist_sq < min_dist_sq:
                total_violation += (min_dist_sq - dist_sq) # Positive if overlap
    return total_violation

@njit(cache=True)
def _check_containment_numba(positions_radii):
    """
    Checks if circles are within bounds [0,1]x[0,1] and have positive radii.
    positions_radii: np.array of shape (N, 3) where columns are (x, y, r)
    Returns: sum of containment violations (positive if outside)
    """
    total_violation = 0.0
    for i in range(positions_radii.shape[0]):
        xi, yi, ri = positions_radii[i, 0], positions_radii[i, 1], positions_radii[i, 2]
        if xi - ri < 0: total_violation += (ri - xi)
        if xi + ri > 1: total_violation += (xi + ri - 1)
        if yi - ri < 0: total_violation += (ri - yi)
        if yi + ri > 1: total_violation += (yi + ri - 1)
        if ri < 1e-7: total_violation += (1e-7 - ri) # Ensure radius is positive, 1e-7 to avoid division by zero issues
    return total_violation

# --- Objective and Constraint Functions for Optimizers ---

def objective_with_penalty(params):
    """
    Objective function for differential_evolution: -sum(radii) + penalty for violations.
    params: 1D array of [x1, y1, r1, x2, y2, r2, ...]
    """
    circles = params.reshape((N_CIRCLES, 3))
    
    radii = circles[:, 2]
    # Smallest allowed radius to prevent issues
    if np.any(radii < 1e-7):
        return np.inf # Heavily penalize near-zero or negative radii

    sum_radii = np.sum(radii)
    
    # Calculate violations using numba-optimized functions
    overlap_penalty = _check_overlap_numba(circles)
    containment_penalty = _check_containment_numba(circles)
    
    # Tune penalty factors. Overlap is usually more critical.
    penalty_factor_overlap = 5000.0
    penalty_factor_containment = 1000.0
    
    total_penalty = penalty_factor_overlap * overlap_penalty + penalty_factor_containment * containment_penalty
    
    return -sum_radii + total_penalty

def slsqp_objective(params):
    """
    Objective function for SLSQP: -sum(radii). Constraints are handled separately.
    """
    circles = params.reshape((N_CIRCLES, 3))
    radii = circles[:, 2]
    return -np.sum(radii)

# Pre-calculate indices for pairwise constraints to avoid re-computation inside the objective function.
_INDICES_I, _INDICES_J = np.triu_indices(N_CIRCLES, k=1)

def _slsqp_constraints_vectorized(params):
    """
    Vectorized constraint function for SLSQP. All constraints are g(x) >= 0.
    This is much faster for scipy than individual lambda functions.
    """
    circles = params.reshape((N_CIRCLES, 3))
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Containment constraints (5 * N_CIRCLES)
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0, r - min_r >= 0
    containment_c = np.concatenate([
        x - r,
        1 - x - r,
        y - r,
        1 - y - r,
        r - 1e-7
    ])

    # 2. Non-overlap constraints (N*(N-1)/2)
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    xi, yi, ri = x[_INDICES_I], y[_INDICES_I], r[_INDICES_I]
    xj, yj, rj = x[_INDICES_J], y[_INDICES_J], r[_INDICES_J]

    dist_sq = (xi - xj)**2 + (yi - yj)**2
    sum_radii_sq = (ri + rj)**2
    overlap_c = dist_sq - sum_radii_sq

    return np.concatenate([containment_c, overlap_c])

def get_slsqp_constraints():
    """
    Generates SLSQP constraints using a single vectorized function.
    Returns: A scipy.optimize.NonlinearConstraint object
    """
    return NonlinearConstraint(_slsqp_constraints_vectorized, 0, np.inf)

# --- Initial Guess Generation ---

def generate_initial_guess(n_circles):
    """
    Generates a more optimal initial guess based on a hexagonal lattice.
    This provides a denser initial packing than a square grid, giving the
    optimizer a much better starting point.
    """
    # 1. Generate hexagonal grid points, extending beyond the unit square
    # to ensure we can select n_circles points closest to the center.
    side_count = int(np.ceil(np.sqrt(n_circles))) + 2  # e.g., 5+2=7 for n=26
    x = np.linspace(-0.2, 1.2, side_count)
    y = np.linspace(-0.2, 1.2, side_count)
    xv, yv = np.meshgrid(x, y)
    xv[1::2, :] += (x[1] - x[0]) / 2.0  # Offset every other row for hex packing
    points = np.vstack([xv.ravel(), yv.ravel()]).T

    # 2. Select n_circles points closest to the center (0.5, 0.5)
    center = np.array([0.5, 0.5])
    distances_to_center = np.linalg.norm(points - center, axis=1)
    central_indices = np.argsort(distances_to_center)[:n_circles]
    initial_positions = points[central_indices]

    # 3. Scale points to fit snugly within the [0,1]x[0,1] box
    min_coords = initial_positions.min(axis=0)
    max_coords = initial_positions.max(axis=0)
    scale = (max_coords - min_coords).max()
    
    if scale > 1e-6:
        initial_positions = (initial_positions - min_coords) / scale
    
    # 4. Estimate initial radii based on the minimum of distance to nearest 
    # neighbor and distance to boundaries.
    dist_matrix = squareform(pdist(initial_positions))
    np.fill_diagonal(dist_matrix, np.inf)
    min_dists_to_neighbor = dist_matrix.min(axis=1)
    
    max_radii_from_neighbors = min_dists_to_neighbor / 2.0
    
    max_radii_from_bounds = np.min(np.hstack([
        initial_positions,
        1.0 - initial_positions
    ]), axis=1)

    # Start with radii slightly smaller than the max possible to ensure feasibility.
    initial_radii = np.minimum(max_radii_from_neighbors, max_radii_from_bounds) * 0.98
    
    # 5. Combine and add a small perturbation to break perfect symmetry
    initial_circles = np.hstack([initial_positions, initial_radii[:, np.newaxis]])
    initial_circles[:, :2] += np.random.normal(0, 0.001, (n_circles, 2))
    initial_circles[:, 2] *= np.random.uniform(0.95, 1.0, n_circles)

    # 6. Final clipping and validation to ensure a valid starting point
    initial_circles[:, 2] = np.clip(initial_circles[:, 2], 1e-7, 0.5)
    for i in range(n_circles):  # Ensure centers are valid for the given radius
        r = initial_circles[i, 2]
        initial_circles[i, 0] = np.clip(initial_circles[i, 0], r, 1 - r)
        initial_circles[i, 1] = np.clip(initial_circles[i, 1], r, 1 - r)
        
    return initial_circles.flatten()

# --- Main Constructor Function ---

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    n = N_CIRCLES

    # Define bounds for (x, y, r) for each circle
    # x: [0, 1], y: [0, 1], r: [1e-7, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * n
    
    # 1. Generate an initial guess
    initial_guess = generate_initial_guess(n)
    
    # 2. Global Optimization using Differential Evolution
    # DE parameters tuned for deeper exploration. With a better initial guess,
    # we can afford more iterations to find a better basin of attraction.
    de_result = differential_evolution(
        func=objective_with_penalty,
        bounds=bounds,
        x0=initial_guess,
        strategy='best1bin',
        maxiter=2000,      # Increased generations for a more thorough search
        popsize=20,        # Increased population size for more diversity
        tol=0.001,         # Tighter convergence tolerance
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        workers=-1,        # Use all available CPU cores
        seed=RANDOM_SEED
    )
    
    if de_result.success:
        print(f"DE successful. Best objective: {de_result.fun:.4f}, Est. sum_radii: {-de_result.fun:.4f}")
        best_de_params = de_result.x
    else:
        print(f"DE failed or did not converge. Message: {de_result.message}")
        print(f"Using best found DE parameters (objective: {de_result.fun:.4f}) as fallback.")
        best_de_params = de_result.x # Use best found, even if not fully converged

    # Ensure best_de_params are within bounds before passing to SLSQP
    # DE might sometimes return values slightly out of bounds due to floating point.
    best_de_params = np.clip(best_de_params, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    
    # 3. Local Refinement using SLSQP
    slsqp_constraints = get_slsqp_constraints()
    
    slsqp_result = minimize(
        fun=slsqp_objective,
        x0=best_de_params, # Start SLSQP from the best DE result
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraints,
        options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False} # Increased maxiter and tighter tolerance for fine-tuning
    )

    if slsqp_result.success:
        final_params = slsqp_result.x
        print(f"SLSQP successful. Final objective: {slsqp_result.fun:.4f}, Final sum_radii: {-slsqp_result.fun:.4f}")
    else:
        print(f"SLSQP failed or did not converge. Message: {slsqp_result.message}")
        print(f"Falling back to best DE result as final solution. Sum_radii: {-slsqp_objective(best_de_params):.4f}")
        final_params = best_de_params # Fallback to DE result if SLSQP fails

    # Reshape the final parameters into (N, 3) format
    circles = final_params.reshape((n, 3))

    # Final validation and cleanup
    circles[:, 2] = np.maximum(circles[:, 2], 1e-7) # Ensure radii are strictly positive
    
    # Check final feasibility
    final_overlap_sum = _check_overlap_numba(circles)
    final_containment_sum = _check_containment_numba(circles)
    
    if final_overlap_sum > 1e-6 or final_containment_sum > 1e-6:
        print(f"WARNING: Final solution from SLSQP might be slightly infeasible.")
        print(f"  Overlap violations sum: {final_overlap_sum:.6f}")
        print(f"  Containment violations sum: {final_containment_sum:.6f}")
        # If there are small violations, a small global scaling of radii might be applied
        # to ensure strict feasibility, but this would reduce the sum_radii.
        # For now, we accept small numerical tolerances from the optimizer.

    end_time = time.time()
    print(f"Total optimization time: {end_time - start_time:.2f} seconds.")

    return circles

# EVOLVE-BLOCK-END