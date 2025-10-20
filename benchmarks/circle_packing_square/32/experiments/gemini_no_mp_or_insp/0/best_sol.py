# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit, prange
import math

# Define constants
NUM_CIRCLES = 32
PENALTY_FACTOR_BOUNDARY = 20000.0  # Increased penalty for boundary violations
PENALTY_FACTOR_OVERLAP = 20000.0   # Increased penalty for overlap violations
MIN_RADIUS = 0.001                  # Minimum allowed radius to avoid numerical issues. Smaller to allow for tiny, filler circles.

@njit
def _calculate_penalties(params: np.ndarray) -> tuple[float, float, float]:
    """
    Numba-optimized function to calculate overlap and boundary penalties.
    This version runs sequentially to avoid numba reduction errors when used
    with multiprocessing-based optimizers like differential_evolution with workers > 1.

    params: 1D array of [x1, y1, r1, ..., xN, yN, rN]
    Returns: (total_sum_radii, total_boundary_penalty, total_overlap_penalty)
    """
    n = NUM_CIRCLES
    total_sum_radii = 0.0
    total_boundary_penalty = 0.0
    total_overlap_penalty = 0.0

    # Pre-calculate radii and x,y for performance and to avoid re-indexing
    radii = np.empty(n, dtype=np.float64)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    
    # This loop is now sequential to prevent numba reduction errors
    for i in range(n):
        xs[i] = params[i*3]
        ys[i] = params[i*3 + 1]
        radii[i] = params[i*3 + 2]
        
        total_sum_radii += radii[i]

        # Boundary penalties
        # Penalize if circle extends beyond unit square [0,1]x[0,1]
        total_boundary_penalty += max(0.0, radii[i] - xs[i])
        total_boundary_penalty += max(0.0, xs[i] + radii[i] - 1.0)
        total_boundary_penalty += max(0.0, radii[i] - ys[i])
        total_boundary_penalty += max(0.0, ys[i] + radii[i] - 1.0)
        
        # Also penalize if radius is too small or negative. Max radius is handled by bounds.
        total_boundary_penalty += max(0.0, MIN_RADIUS - radii[i])


    # Overlap penalties
    # Iterate over unique pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            
            # Use math.sqrt for scalar in numba
            dist_sq = dx*dx + dy*dy
            
            # Calculate distance between centers
            dist = math.sqrt(dist_sq) # math.sqrt(0.0) is 0.0, so no special handling needed for dist_sq=0.0
            
            required_dist = radii[i] + radii[j]
            
            # If circles overlap, add to penalty
            total_overlap_penalty += max(0.0, required_dist - dist)

    return total_sum_radii, total_boundary_penalty, total_overlap_penalty

def objective_function_de(params: np.ndarray) -> float:
    """
    Objective function for differential_evolution. Maximizes sum of radii, penalizes violations.
    """
    sum_radii, boundary_penalty, overlap_penalty = _calculate_penalties(params)
    
    # We want to maximize sum_radii, so we minimize -sum_radii
    # Add penalties to make infeasible solutions undesirable
    objective_value = -sum_radii \
                      + PENALTY_FACTOR_BOUNDARY * boundary_penalty \
                      + PENALTY_FACTOR_OVERLAP * overlap_penalty
    
    return objective_value

@njit
def objective_function_slsqp(params_local: np.ndarray) -> float:
    """
    Pure objective function for SLSQP (only -sum_radii), as constraints are handled separately.
    """
    total_sum_radii = 0.0
    for i in prange(NUM_CIRCLES):
        total_sum_radii += params_local[i*3 + 2]
    return -total_sum_radii

@njit
def all_inequality_constraints(params_local: np.ndarray) -> np.ndarray:
    """
    Returns an array of all inequality constraint values (g(x) >= 0) for SLSQP.
    """
    n = NUM_CIRCLES
    n_constraints_per_circle = 5 # x-r, 1-x-r, y-r, 1-y-r, r-min_r
    n_overlap_constraints = n * (n - 1) // 2
    total_constraints = n * n_constraints_per_circle + n_overlap_constraints
    
    c_values = np.empty(total_constraints, dtype=np.float64)
    c_idx = 0
    
    # Boundary constraints
    for i in range(n):
        x, y, r = params_local[i*3], params_local[i*3+1], params_local[i*3+2]
        c_values[c_idx] = x - r              # x - r >= 0
        c_idx += 1
        c_values[c_idx] = 1.0 - x - r        # 1 - x - r >= 0
        c_idx += 1
        c_values[c_idx] = y - r              # y - r >= 0
        c_idx += 1
        c_values[c_idx] = 1.0 - y - r        # 1 - y - r >= 0
        c_idx += 1
        c_values[c_idx] = r - MIN_RADIUS     # r - MIN_RADIUS >= 0
        c_idx += 1

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = params_local[i*3] - params_local[j*3]
            dy = params_local[i*3+1] - params_local[j*3+1]
            # Use squared distance for numerical stability with gradient-based optimizers
            dist_sq = dx*dx + dy*dy
            required_dist = params_local[i*3+2] + params_local[j*3+2]
            required_dist_sq = required_dist * required_dist
            c_values[c_idx] = dist_sq - required_dist_sq # dist_sq - (ri + rj)^2 >= 0
            c_idx += 1
    return c_values

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = NUM_CIRCLES
    popsize = 150 # Increased popsize for more diverse initial population and DE
    dims = n * 3
    
    # Define bounds for x, y, r for differential_evolution and SLSQP
    # x, y are in [0, 1]
    # r is in [MIN_RADIUS, 0.5] (0.5 is max possible radius for a single circle)
    bounds = [(0.0, 1.0), (0.0, 1.0), (MIN_RADIUS, 0.5)] * n

    # --- Create a structured initial population for DE ---
    # This guides the optimizer towards a good region of the search space from the start.
    initial_population = np.zeros((popsize, dims))
    rng = np.random.default_rng(seed=42) # Ensure determinism

    # Helper to create a square grid solution
    def _create_grid_solution(num_circles: int, grid_x: int, grid_y: int, radius_factor: float = 0.98) -> tuple[np.ndarray, float, float, float]:
        sol = np.zeros(num_circles * 3)
        dx, dy = 1.0 / grid_x, 1.0 / grid_y
        r_grid = min(dx, dy) / 2.0 * radius_factor # Use 98% of max possible radius for a grid cell
        
        idx = 0
        for i in range(grid_x):
            for j in range(grid_y):
                if idx < num_circles:
                    sol[idx*3]     = (i + 0.5) * dx
                    sol[idx*3 + 1] = (j + 0.5) * dy
                    sol[idx*3 + 2] = r_grid
                    idx += 1
        return sol, r_grid, dx, dy

    # Helper to create a hexagonal grid solution
    def _create_hex_solution(num_circles: int, r_target: float, radius_factor: float = 0.98) -> np.ndarray:
        sol = np.zeros(num_circles * 3)
        r_hex = r_target * radius_factor
        
        # Max number of rows
        max_rows = int(np.floor((1.0 - r_hex - r_hex) / (r_hex * np.sqrt(3)))) + 1
        if max_rows < 1: max_rows = 1
        
        idx = 0
        for row in range(max_rows):
            y_center = r_hex + row * r_hex * np.sqrt(3)
            
            # If y_center is too close to 1.0, break
            if y_center + r_hex > 1.0 + 1e-9: # Add small tolerance
                break
            
            # Alternate row offset
            x_start_offset = r_hex if row % 2 == 0 else 2 * r_hex # First circle in row 0 is at r, in row 1 at 2r
            
            # Calculate max columns for this specific row, considering its offset
            num_cols_this_row = int(np.floor((1.0 - x_start_offset - r_hex) / (2 * r_hex))) + 1
            if num_cols_this_row < 1: num_cols_this_row = 1

            for col in range(num_cols_this_row):
                x_center = x_start_offset + col * 2 * r_hex
                
                if x_center + r_hex > 1.0 + 1e-9: # Add small tolerance
                    break
                
                if idx < num_circles:
                    sol[idx*3]     = x_center
                    sol[idx*3 + 1] = y_center
                    sol[idx*3 + 2] = r_hex
                    idx += 1
                else:
                    break # Reached num_circles
            
            if idx >= num_circles:
                break # Reached num_circles
        
        # Fill any remaining circles if we didn't reach num_circles
        # (e.g. if r_target was too large for the square)
        while idx < num_circles:
            sol[idx*3] = rng.uniform(MIN_RADIUS, 1.0 - MIN_RADIUS)
            sol[idx*3 + 1] = rng.uniform(MIN_RADIUS, 1.0 - MIN_RADIUS)
            sol[idx*3 + 2] = MIN_RADIUS
            idx += 1
        
        return sol

    # Helper to create a solution with large corner circles and random fillers
    def _create_corner_fill_solution(num_circles: int, corner_r: float, filler_r_base: float) -> np.ndarray:
        sol = np.zeros(num_circles * 3)
        idx = 0
        
        # Place 4 corner circles
        if idx < num_circles:
            sol[idx*3:idx*3+3] = [corner_r, corner_r, corner_r]
            idx += 1
        if idx < num_circles:
            sol[idx*3:idx*3+3] = [1.0 - corner_r, corner_r, corner_r]
            idx += 1
        if idx < num_circles:
            sol[idx*3:idx*3+3] = [corner_r, 1.0 - corner_r, corner_r]
            idx += 1
        if idx < num_circles:
            sol[idx*3:idx*3+3] = [1.0 - corner_r, 1.0 - corner_r, corner_r]
            idx += 1
        
        # Fill the remaining space with randomly placed smaller circles
        remaining_circles = num_circles - idx
        for k in range(remaining_circles):
            r_fill = rng.uniform(MIN_RADIUS, filler_r_base) # Vary filler radii
            # Place randomly within the square, ensuring radius fits
            sol[(idx+k)*3] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 1] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 2] = r_fill
        
        return sol

    # Helper to create a solution with a large central circle and random fillers
    def _create_central_fill_solution(num_circles: int, central_r: float, filler_r_base: float) -> np.ndarray:
        sol = np.zeros(num_circles * 3)
        idx = 0
        
        # Place one central circle
        if idx < num_circles:
            sol[idx*3:idx*3+3] = [0.5, 0.5, central_r]
            idx += 1
        
        # Fill remaining with random smaller circles
        remaining_circles = num_circles - idx
        for k in range(remaining_circles):
            r_fill = rng.uniform(MIN_RADIUS, filler_r_base)
            sol[(idx+k)*3] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 1] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 2] = r_fill
        
        return sol

    # Helper to create a solution with circles hugging the boundary
    def _create_boundary_hug_solution(num_circles: int, num_boundary: int, boundary_r: float, filler_r_base: float) -> np.ndarray:
        sol = np.zeros(num_circles * 3)
        idx = 0
        
        # Place circles along y-axis (x=0 edge), including the corner
        num_on_y = num_boundary // 2
        for i in range(num_on_y):
            y_center = boundary_r + i * 2 * boundary_r
            if y_center + boundary_r > 1.0 or idx >= num_circles:
                break
            sol[idx*3:idx*3+3] = [boundary_r, y_center, boundary_r]
            idx += 1

        # Place circles along x-axis (y=0 edge), skipping the corner
        num_on_x = num_boundary - num_on_y
        for i in range(num_on_x):
            x_center = boundary_r + (i + 1) * 2 * boundary_r # Start at 3r, 5r, ...
            if x_center + boundary_r > 1.0 or idx >= num_circles:
                break
            sol[idx*3:idx*3+3] = [x_center, boundary_r, boundary_r]
            idx += 1

        # Fill the remaining space with randomly placed smaller circles
        remaining_circles = num_circles - idx
        for k in range(remaining_circles):
            r_fill = rng.uniform(MIN_RADIUS, filler_r_base) # Vary filler radii
            sol[(idx+k)*3] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 1] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 2] = r_fill
        
        return sol

    # Helper to create a solution with circles packing all four boundaries
    def _create_full_boundary_solution(num_circles: int, boundary_r: float, filler_r_base: float) -> np.ndarray:
        sol = np.zeros(num_circles * 3)
        idx = 0
        
        # How many circles fit on one edge
        num_per_edge = int(np.floor(1.0 / (2 * boundary_r)))
        
        # Place on bottom edge (y=r)
        for i in range(num_per_edge):
            if idx >= num_circles: break
            x = boundary_r + i * 2 * boundary_r
            if x + boundary_r > 1.0 + 1e-9: break
            sol[idx*3:idx*3+3] = [x, boundary_r, boundary_r]
            idx += 1
        
        # Place on top edge (y=1-r)
        for i in range(num_per_edge):
            if idx >= num_circles: break
            x = boundary_r + i * 2 * boundary_r
            if x + boundary_r > 1.0 + 1e-9: break
            sol[idx*3:idx*3+3] = [x, 1.0 - boundary_r, boundary_r]
            idx += 1
            
        # Place on left edge (x=r), skipping corners
        num_vertical_inner = int(np.floor((1.0 - 4 * boundary_r) / (2 * boundary_r)))
        if num_vertical_inner < 0: num_vertical_inner = 0

        for i in range(num_vertical_inner):
            if idx >= num_circles: break
            y = 3 * boundary_r + i * 2 * boundary_r
            if y + boundary_r > 1.0 - boundary_r + 1e-9: break
            sol[idx*3:idx*3+3] = [boundary_r, y, boundary_r]
            idx += 1
            
        # Place on right edge (x=1-r), skipping corners
        for i in range(num_vertical_inner):
            if idx >= num_circles: break
            y = 3 * boundary_r + i * 2 * boundary_r
            if y + boundary_r > 1.0 - boundary_r + 1e-9: break
            sol[idx*3:idx*3+3] = [1.0 - boundary_r, y, boundary_r]
            idx += 1

        # Fill remaining with random smaller circles
        remaining_circles = num_circles - idx
        for k in range(remaining_circles):
            r_fill = rng.uniform(MIN_RADIUS, filler_r_base)
            sol[(idx+k)*3] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 1] = rng.uniform(r_fill, 1.0 - r_fill)
            sol[(idx+k)*3 + 2] = r_fill
        
        return sol


    # Base solutions for initial population
    base_solutions_list = []

    # 1. Square grids
    # 6x6 grid (36 cells, take 32)
    sol, r, dx, dy = _create_grid_solution(n, 6, 6)
    base_solutions_list.append((sol, r, dx))
    
    # 7x5 grid (35 cells, take 32)
    sol, r, dx, dy = _create_grid_solution(n, 7, 5)
    base_solutions_list.append((sol, r, dx))

    # 5x7 grid (35 cells, take 32)
    sol, r, dx, dy = _create_grid_solution(n, 5, 7)
    base_solutions_list.append((sol, r, dx))

    # 8x4 grid (32 cells exactly)
    sol, r, dx, dy = _create_grid_solution(n, 8, 4)
    base_solutions_list.append((sol, r, dx))

    # 2. Hexagonal grids
    # Target radius for hexagonal packing, derived from ideal square packing for 32 circles: 0.5/sqrt(32) approx 0.088
    # Try a few slightly different radii for diversity
    hex_r_targets = [0.070, 0.075, 0.080, 0.085, 0.088, 0.090, 0.095, 0.100, 0.105] # Expanded range
    for r_target in hex_r_targets:
        sol = _create_hex_solution(n, r_target)
        r_hex_est = np.mean(sol[2::3]) if np.any(sol[2::3] > MIN_RADIUS) else r_target
        base_solutions_list.append((sol, r_hex_est, 2 * r_hex_est)) # dx_ref for hex is roughly 2r

    # 3. Strategic placement solutions (multi-scale idea)
    # Corner-filled solutions
    corner_radii_options = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25] # Expanded radii for the corner circles, added 0.25
    filler_radii_options = [0.04, 0.05, 0.06] # Base radii for the filler circles
    for cr in corner_radii_options:
        for fr in filler_radii_options:
            sol = _create_corner_fill_solution(n, cr, fr)
            avg_r = np.mean(sol[2::3]) if np.any(sol[2::3] > MIN_RADIUS) else (cr + fr) / 2
            base_solutions_list.append((sol, avg_r, avg_r * 2))

    # Central-filled solutions
    central_radii_options = [0.18, 0.20, 0.22, 0.25, 0.28] # Expanded radii for the central circle
    filler_radii_options_central = [0.03, 0.04, 0.05] # Base radii for the filler circles
    for cr_cen in central_radii_options:
        for fr_cen in filler_radii_options_central:
            sol = _create_central_fill_solution(n, cr_cen, fr_cen)
            avg_r = np.mean(sol[2::3]) if np.any(sol[2::3] > MIN_RADIUS) else (cr_cen + fr_cen) / 2
            base_solutions_list.append((sol, avg_r, avg_r * 2))

    # 4. Boundary-hugging solutions (2-sided)
    boundary_radii_options = [0.08, 0.1, 0.12]
    num_boundary_options = [8, 10, 12]
    filler_radii_options_boundary = [0.04, 0.05]
    for br in boundary_radii_options:
        for nb in num_boundary_options:
            for fr_b in filler_radii_options_boundary:
                sol = _create_boundary_hug_solution(n, nb, br, fr_b)
                avg_r = np.mean(sol[2::3])
                base_solutions_list.append((sol, avg_r, avg_r * 2))

    # 5. Full boundary solutions (4-sided)
    full_boundary_radii = [0.06, 0.07, 0.08]
    filler_radii_full_boundary = [0.03, 0.04]
    for br_full in full_boundary_radii:
        for fr_full in filler_radii_full_boundary:
            sol = _create_full_boundary_solution(n, br_full, fr_full)
            avg_r = np.mean(sol[2::3])
            base_solutions_list.append((sol, avg_r, avg_r * 2))

    num_base_solutions = len(base_solutions_list)

    # Fill initial_population from base solutions and their perturbed variants
    for i in range(popsize):
        # First fill with the direct base solutions
        if i < num_base_solutions:
            initial_population[i] = base_solutions_list[i][0]
        else:
            # Create variants by adding progressively larger noise, cycling through base solutions
            # Ensure noise_factor is not zero if popsize > num_base_solutions
            noise_factor = (i - num_base_solutions + 1) / (popsize - num_base_solutions) if popsize > num_base_solutions else 0.5 
            
            base_idx = (i - num_base_solutions) % num_base_solutions
            base_to_perturb, r_init_ref, dx_ref = base_solutions_list[base_idx]

            # Scale noise based on reference dimensions of the pattern
            noise_pos_std = noise_factor * dx_ref * 0.25
            noise_rad_std = noise_factor * r_init_ref * 0.25
            
            noise = np.empty(dims)
            for j in range(n):
                noise[j*3]   = rng.normal(scale=noise_pos_std)
                noise[j*3+1] = rng.normal(scale=noise_pos_std)
                noise[j*3+2] = rng.normal(scale=noise_rad_std)
            
            perturbed_solution = base_to_perturb + noise
            
            # Clip to bounds
            for j in range(n):
                perturbed_solution[j*3]   = np.clip(perturbed_solution[j*3],   0.0, 1.0)
                perturbed_solution[j*3+1] = np.clip(perturbed_solution[j*3+1], 0.0, 1.0)
                perturbed_solution[j*3+2] = np.clip(perturbed_solution[j*3+2], MIN_RADIUS, 0.5)
            initial_population[i] = perturbed_solution

    # --- Phase 1: Global Optimization with Differential Evolution ---
    de_result = differential_evolution(
        objective_function_de,
        bounds,
        init=initial_population, # Use the structured initial population
        strategy='best2bin',      # More robust strategy
        popsize=popsize,          # Use the variable defined above (increased to 150)
        maxiter=10000,            # Keep maxiter high, will terminate on tol
        recombination=0.9,        # Standard robust setting
        mutation=(0.5, 1.0),      # Standard robust setting
        tol=1e-6,                 # Tighter tolerance for more precise global optimization
        polish=False,             # Disable internal polishing, we do a more robust one later
        seed=42,
        workers=-1,
        disp=False
    )
    
    best_params_de = de_result.x
    
    # --- Phase 2: Local Refinement with SLSQP ---
    slsqp_constraints = {'type': 'ineq', 'fun': all_inequality_constraints}

    slsqp_result = minimize(
        objective_function_slsqp,
        best_params_de,
        method='SLSQP',
        bounds=bounds,
        constraints=slsqp_constraints,
        options={'maxiter': 10000, 'ftol': 1e-9, 'disp': False} # Increased maxiter for deeper local refinement
    )
    
    # Use the result from SLSQP if it converged successfully, otherwise fall back to DE's best result
    if slsqp_result.success:
        optimal_params = slsqp_result.x
    else:
        optimal_params = best_params_de

    # --- Phase 3: Multiple Local Refinements with Perturbations (Iterated Local Search) ---
    best_overall_params = optimal_params
    best_overall_sum_radii = -objective_function_slsqp(optimal_params) # Note the negation

    num_refinements = 30 # Increased number of refinement attempts
    initial_perturbation_scale = 0.010 # Start with a slightly larger perturbation
    final_perturbation_scale = 0.001   # End with a fine-tuning perturbation

    for i_refine in range(num_refinements):
        # Annealing-like decay for perturbation scale
        decay_factor = (1 - i_refine / num_refinements)
        current_pos_scale = final_perturbation_scale + (initial_perturbation_scale - final_perturbation_scale) * decay_factor**2
        current_rad_scale = current_pos_scale / 4.0 # Radii are more sensitive

        perturbed_params = best_overall_params.copy()
        
        # Apply small random noise to positions and radii
        noise = rng.normal(scale=1.0, size=perturbed_params.shape)
        for i in range(n):
            noise[i*3]   *= current_pos_scale
            noise[i*3+1] *= current_pos_scale
            noise[i*3+2] *= current_rad_scale
        
        perturbed_params += noise
        
        # Clip to bounds to ensure feasibility
        for j in range(n):
            perturbed_params[j*3]   = np.clip(perturbed_params[j*3],   0.0, 1.0)
            perturbed_params[j*3+1] = np.clip(perturbed_params[j*3+1], 0.0, 1.0)
            perturbed_params[j*3+2] = np.clip(perturbed_params[j*3+2], MIN_RADIUS, 0.5)

        slsqp_result_refine = minimize(
            objective_function_slsqp,
            perturbed_params,
            method='SLSQP',
            bounds=bounds,
            constraints=slsqp_constraints,
            options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False} # More iterations and tighter tolerance for refinement
        )
        
        if slsqp_result_refine.success:
            current_sum_radii = -objective_function_slsqp(slsqp_result_refine.x)
            # Only accept a new solution if it's significantly better, to avoid numerical drift
            if current_sum_radii > best_overall_sum_radii + 1e-8:
                best_overall_sum_radii = current_sum_radii
                best_overall_params = slsqp_result_refine.x

    optimal_params = best_overall_params

    # Reshape the optimal parameters into the (N, 3) format
    optimal_circles = optimal_params.reshape((n, 3))
    
    return optimal_circles

# EVOLVE-BLOCK-END
