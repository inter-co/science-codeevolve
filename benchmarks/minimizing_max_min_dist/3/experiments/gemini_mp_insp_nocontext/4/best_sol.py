# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize, minimize_scalar # Added minimize_scalar for h pre-optimization
from numba import njit # Changed from numba to njit for clarity and direct usage

# Numba-accelerated core objective function (adapted from Inspiration 3)
@njit(fastmath=True, cache=True)
def _objective_core_numba(coordinates_flat: np.ndarray, n_points: int, dim: int) -> float:
    """
    Numba-accelerated core objective function for Cartesian coordinates.
    Calculates -dmin/dmax after centering and scaling points to a unit sphere.
    Returns np.inf for degenerate configurations to guide the optimizer away.
    """
    points = coordinates_flat.reshape((n_points, dim))

    # 1. Centering
    centroid = np.empty(dim, dtype=np.float64)
    for d_idx in range(dim):
        sum_dim = 0.0
        for i in range(n_points):
            sum_dim += points[i, d_idx]
        centroid[d_idx] = sum_dim / n_points
    
    centered_points = np.empty_like(points, dtype=np.float64)
    for i in range(n_points):
        for d_idx in range(dim):
            centered_points[i, d_idx] = points[i, d_idx] - centroid[d_idx]
    
    # 2. Scaling to fit within a unit sphere
    distances_from_origin = np.empty(n_points, dtype=np.float64)
    max_dist_from_origin = 0.0
    for i in range(n_points):
        norm_sq = 0.0
        for d_idx in range(dim):
            norm_sq += centered_points[i, d_idx]**2
        distances_from_origin[i] = np.sqrt(norm_sq)
        if distances_from_origin[i] > max_dist_from_origin:
            max_dist_from_origin = distances_from_origin[i]

    # Handle edge case where all points are coincident or nearly so.
    if max_dist_from_origin < 1e-9:
        return np.inf # Return a large positive number to penalize heavily for minimization

    normalized_points = np.empty_like(points, dtype=np.float64)
    for i in range(n_points):
        # Avoid division by zero if max_dist_from_origin is still tiny despite the check above
        if max_dist_from_origin < 1e-12:
            # If all points are at the center, they are coincident, so assign them to 0.
            normalized_points[i, :] = 0.0
        else:
            for d_idx in range(dim):
                normalized_points[i, d_idx] = centered_points[i, d_idx] / max_dist_from_origin

    # 3. Calculate all pairwise Euclidean distances (manual implementation for Numba)
    num_distances = n_points * (n_points - 1) // 2
    
    if num_distances == 0: # Handle cases with less than 2 points
        return np.inf

    distances_array = np.empty(num_distances, dtype=np.float64)
    k = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dx = normalized_points[i, 0] - normalized_points[j, 0]
            dy = normalized_points[i, 1] - normalized_points[j, 1]
            dz = normalized_points[i, 2] - normalized_points[j, 2]
            distances_array[k] = np.sqrt(dx**2 + dy**2 + dz**2)
            k += 1
    
    # If all points collapse after normalization (e.g., if there's only one effective point)
    if np.all(distances_array < 1e-12): # Check for extremely small distances
        return np.inf # Penalize heavily

    # 4. Find min and max distances and compute objective
    dmin = distances_array[0] # Initialize with first element
    dmax = distances_array[0] # Initialize with first element
    for d_val in distances_array:
        if d_val < dmin:
            dmin = d_val
        if d_val > dmax:
            dmax = d_val

    # Safeguard: if dmax is still very small, it implies a degenerate configuration
    if dmax < 1e-10:
        return np.inf # Penalize heavily

    # Return the negative ratio because optimizers minimize, and we want to maximize the ratio.
    return -dmin / dmax

# Wrapper function for the Numba-accelerated core objective
def _objective_func(coordinates_flat: np.ndarray, n_points: int, dim: int) -> float:
    return _objective_core_numba(coordinates_flat, n_points, dim)

# Function to generate bicapped hexagonal antiprism (adapted from Inspiration 3)
def generate_bicapped_hex_antiprism_cartesian(h: float, n_ring: int = 6) -> np.ndarray:
    """Generates Cartesian coordinates for a bicapped hexagonal antiprism on a unit sphere."""
    # Two poles (the caps)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    # Radius of the hexagonal rings, derived from h so points are on the unit sphere
    # Clip h to avoid sqrt of negative number if h is slightly > 1 due to numerical issues
    r = np.sqrt(1 - np.clip(h, -1.0, 1.0)**2)

    # Top ring of 6 points
    thetas_top = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    x_top = r * np.cos(thetas_top)
    y_top = r * np.sin(thetas_top)
    z_top = np.full(n_ring, h)
    top_ring = np.vstack([x_top, y_top, z_top]).T

    # Bottom ring of 6 points, twisted by pi/6 relative to the top ring
    twist_angle = np.pi / n_ring
    thetas_bottom = thetas_top + twist_angle
    x_bottom = r * np.cos(thetas_bottom)
    y_bottom = r * np.sin(thetas_bottom)
    z_bottom = np.full(n_ring, -h)
    bottom_ring = np.vstack([x_bottom, y_bottom, z_bottom]).T
    
    # Combine all 14 points
    points = np.vstack([poles, top_ring, bottom_ring])
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    This function uses a hybrid optimization strategy combining the best elements:
    1.  **Domain and Normalization**: Points are optimized in Cartesian space (e.g., [-1,1]³),
        then centered and scaled to a unit sphere *inside the Numba-jitted objective function*.
        This frees the optimizer from explicit spherical constraints.
    2.  **Pre-optimization of Antiprism 'h'**: The optimal height parameter 'h' for the bicapped
        hexagonal antiprism is found using scalar optimization, providing a superior initial guess.
    3.  **Hybrid Multi-start Optimization**: Multiple runs of `dual_annealing` (global search)
        are performed, each immediately followed by a `minimize` (local refinement) step.
        This aggressive strategy thoroughly explores promising basins.
    4.  **Numba Acceleration**: Critical distance calculations and objective function logic are
        Numba-jitted for maximum performance.

    Returns:
        points: np.ndarray of shape (14, 3) containing the Cartesian (x,y,z) coordinates of the 14 points,
                normalized to lie on the surface of a unit sphere.
    """
    n_points = 14
    d = 3
    n_dims = n_points * d

    base_seed = 42 # Base seed for overall reproducibility

    # Define bounds for Cartesian coordinates (e.g., a cube of side 2, centered at origin)
    bounds = [(-1.0, 1.0)] * n_dims

    best_overall_objective = np.inf
    best_overall_x = None

    # --- Pre-optimization of Bicapped Hexagonal Antiprism Parameter 'h' (from Inspiration 2) ---
    # The objective for scalar optimization of 'h' will use the _objective_func
    # after converting the antiprism points to a flattened Cartesian array.
    def _objective_h(h_val: float) -> float:
        # Generate antiprism points for the given h. These are already on a unit sphere.
        antiprism_points = generate_bicapped_hex_antiprism_cartesian(h_val)
        # Flatten and pass to the main objective function.
        # The main objective will center and scale them, but since they are already on a unit sphere,
        # centering will move the centroid to origin, and scaling will effectively scale by 1.
        return _objective_func(antiprism_points.flatten(), n_points, d)
    
    # Search for optimal h in a reasonable range, e.g., (0, 1).
    # Initial guess for h can be 0.5 or 1/sqrt(3) which is approx 0.577.
    h_opt_result = minimize_scalar(_objective_h, bounds=(0.01, 0.99), method='bounded')
    optimal_h = h_opt_result.x
    
    # Generate the bicapped hexagonal antiprism as the first, high-quality initial guess, using optimal_h
    initial_antiprism_x0 = generate_bicapped_hex_antiprism_cartesian(h=optimal_h).flatten()
    # --- End Pre-optimization ---

    # Tuned parameters for multi-start optimization, balancing exploration and time budget.
    # Increased num_restarts and iterations to maximize search within the time limit,
    # learning from aggressive strategies in inspiration programs.
    num_restarts = 10 # Number of global + local optimization runs
    maxiter_dual_annealing = 2500 # Iterations for global search per restart
    maxiter_minimize = 1500 # Iterations for local refinement per restart

    for i in range(num_restarts):
        current_run_seed = base_seed + i
        current_rng = np.random.default_rng(current_run_seed) # Dedicated RNG for random initial points

        if i == 0: # First run uses the strong, geometrically-informed antiprism guess
            initial_guess_x = initial_antiprism_x0
        else: # Subsequent runs use random uniform points within the bounds for diverse exploration
            initial_guess_x = current_rng.uniform(low=-1.0, high=1.0, size=n_dims)

        # --- Global Search Phase (dual_annealing) ---
        result_sa = dual_annealing(
            func=_objective_func,
            bounds=bounds,
            args=(n_points, d),
            x0=initial_guess_x,
            seed=current_run_seed,
            maxiter=maxiter_dual_annealing,
            no_local_search=True, # Disable internal local search to focus on global exploration
        )
        
        # --- Local Refinement Phase (minimize) for EACH restart ---
        # Starting from the best point found by this dual_annealing run.
        result_minimize = minimize(
            fun=_objective_func,
            x0=result_sa.x, # Start local search from the best global solution of this restart
            args=(n_points, d),
            bounds=bounds,
            method='L-BFGS-B', # A robust local optimization method that supports bounds
            options={'maxiter': maxiter_minimize, 'ftol': 1e-10, 'gtol': 1e-8}, # Tighter tolerances
        )

        # Update the best overall solution found so far
        if result_minimize.fun < best_overall_objective:
            best_overall_objective = result_minimize.fun
            best_overall_x = result_minimize.x.copy()

    # Handle the unlikely case where no valid solution was found after all restarts
    if best_overall_x is None:
        # Fallback to the pre-optimized bicapped antiprism if all multi-starts fail
        final_optimized_x_flat = initial_antiprism_x0
    else:
        final_optimized_x_flat = best_overall_x

    # Reshape and apply the final normalization to the optimized points for output.
    # The _objective_core_numba already performs centering and scaling internally.
    # We call it one last time explicitly here to ensure the *output* points are precisely
    # centered at origin and scaled to fit within a unit sphere, as required by the problem.
    final_points = final_optimized_x_flat.reshape((n_points, d))
    
    centroid = np.mean(final_points, axis=0)
    centered_points = final_points - centroid
    
    distances_from_origin = np.linalg.norm(centered_points, axis=1)
    max_dist_from_origin = np.max(distances_from_origin)

    # Safeguard against max_dist_from_origin being zero or near-zero for the final result.
    if max_dist_from_origin < 1e-9:
        # If points collapsed, return a trivial configuration (e.g., all points at origin).
        return np.zeros((n_points, d))
        
    normalized_optimized_points = centered_points / max_dist_from_origin

    return normalized_optimized_points
# EVOLVE-BLOCK-END