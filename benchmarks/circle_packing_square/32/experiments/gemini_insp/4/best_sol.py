# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, differential_evolution, NonlinearConstraint # Added differential_evolution, NonlinearConstraint
from numba import njit
from scipy.stats import qmc # Added for Latin Hypercube Sampling initial population

# --- Constants ---
N_CIRCLES = 32
MAX_POSSIBLE_RADIUS = 0.5
MIN_RADIUS_EPSILON = 1e-7 # Added from Inspiration 1

# Numba JIT-compiled function for faster constraint evaluation
@njit
def _evaluate_constraints_numba(params: np.ndarray, n: int) -> np.ndarray:
    """
    Evaluates all inequality constraints for circle packing.
    Constraints:
    1. Circle containment within the unit square: ri <= xi <= 1-ri, ri <= yi <= 1-ri
    2. Non-overlap between circles: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj

    Parameters:
        params: 1D numpy array of shape (3*n,) containing [x1..xn, y1..yn, r1..rn]
        n: Number of circles

    Returns:
        A 1D numpy array of constraint values. All values must be >= 0 for a feasible solution.
    """
    xs = params[0:n]
    ys = params[n:2*n]
    rs = params[2*n:3*n]

    num_boundary_constraints = 4 * n
    num_overlap_constraints = n * (n - 1) // 2
    total_constraints = num_boundary_constraints + num_overlap_constraints
    
    constraints = np.empty(total_constraints, dtype=np.float64)
    idx = 0

    # 1. Boundary constraints (4*N constraints)
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    for i in range(n):
        constraints[idx] = xs[i] - rs[i]
        idx += 1
        constraints[idx] = 1.0 - xs[i] - rs[i]
        idx += 1
        constraints[idx] = ys[i] - rs[i]
        idx += 1
        constraints[idx] = 1.0 - ys[i] - rs[i]
        idx += 1

    # 2. Non-overlap constraints (N*(N-1)/2 constraints)
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (rs[i] + rs[j])**2
            constraints[idx] = dist_sq - min_dist_sq
            idx += 1
    
    return constraints

def _objective(params: np.ndarray, n: int) -> float:
    """
    Objective function to minimize: negative sum of radii.
    """
    return -np.sum(params[2*n:3*n])

# Helper function to generate a rectangular grid initial guess (adapted from Insp 1)
def _generate_initial_grid_guess(n_circles, rows, cols, random_perturbation=0.01, seed=None):
    """
    Generates a single initial guess vector based on a grid.
    """
    if seed is not None:
        np.random.seed(seed)

    r_base = 0.5 / max(rows, cols)
    x_coords = np.linspace(r_base, 1 - r_base, cols)
    y_coords = np.linspace(r_base, 1 - r_base, rows)
    
    initial_x, initial_y, initial_r = [], [], []
    count = 0
    for j in range(rows):
        for i in range(cols):
            if count < n_circles:
                x_val = x_coords[i] + np.random.uniform(-random_perturbation, random_perturbation)
                y_val = y_coords[j] + np.random.uniform(-random_perturbation, random_perturbation)
                r_val = r_base * (1.0 + np.random.uniform(-0.1, 0.1))
                initial_x.append(np.clip(x_val, MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON))
                initial_y.append(np.clip(y_val, MIN_RADIUS_EPSILON, 1 - MIN_RADIUS_EPSILON))
                initial_r.append(np.clip(r_val, MIN_RADIUS_EPSILON, MAX_POSSIBLE_RADIUS))
                count += 1
    
    return np.concatenate([np.array(initial_x), np.array(initial_y), np.array(initial_r)])


# Helper function to generate a hexagonal grid initial guess, adapted from Inspiration 1
def _generate_hexagonal_guess(n_circles):
    """
    Generates an initial guess based on a quasi-hexagonal packing for exactly 32 circles.
    Adapted from Inspiration Program 1.
    """
    if n_circles != 32:
        # Fallback to a grid guess if not 32 circles, though this function is specific.
        return _generate_initial_grid_guess(n_circles, 6, 6, random_perturbation=0.01)

    n_circles_per_row = [6, 5, 6, 5, 6, 4] # Sums to 32

    # Calculate initial radius based on the tighter horizontal constraint for 6 circles in a row.
    r_initial_base = 1.0 / 12.0 # 6 circles, centers at r, 3r, ..., 11r. Total width 12r. So 12*r <= 1 => r <= 1/12.

    initial_xs_list = []
    initial_ys_list = []

    current_y_pos = r_initial_base # Start first row center at r_initial_base
    y_spacing = r_initial_base * np.sqrt(3) # Hexagonal vertical spacing (distance between row centers)
    
    for i, num_in_row in enumerate(n_circles_per_row):
        # Calculate x-offset for alternating rows for hexagonal packing.
        x_row_offset = 0.0
        if i % 2 == 1: 
            x_row_offset = r_initial_base 

        # Generate x-coordinates for circles in the current row assuming tight packing.
        x_centers_relative = np.array([(2*k + 1) * r_initial_base for k in range(num_in_row)])
        
        # Calculate the total span of the circles in this row.
        current_row_total_width = (2 * num_in_row) * r_initial_base
        
        # Calculate the initial shift needed to center the *un-offset* row horizontally within [0,1].
        x_shift_unoffset = (1.0 - current_row_total_width) / 2.0
        
        # Apply this centering shift and then the hexagonal offset.
        x_row_centers = x_centers_relative + x_shift_unoffset + x_row_offset
        
        initial_xs_list.extend(x_row_centers)
        initial_ys_list.extend(np.full(num_in_row, current_y_pos))
        
        # Update y position for the next row
        current_y_pos += y_spacing
    
    # Convert lists to numpy arrays
    initial_xs = np.array(initial_xs_list)
    initial_ys_uncentered = np.array(initial_ys_list)
    initial_rs = np.full(n_circles, r_initial_base)

    # --- Vertical Centering of the entire initial packing ---
    y_min_boundary = initial_ys_uncentered.min() - r_initial_base
    y_max_boundary = initial_ys_uncentered.max() + r_initial_base
    total_packing_height = y_max_boundary - y_min_boundary
    
    y_overall_shift = (1.0 - total_packing_height) / 2.0 - y_min_boundary
    
    initial_ys = initial_ys_uncentered + y_overall_shift

    # Combine into a single parameter vector
    x0_base = np.concatenate((initial_xs, initial_ys, initial_rs))

    # Add a small random perturbation to break symmetry and aid optimization.
    # Using a fixed seed for reproducibility.
    rng = np.random.default_rng(42) 
    perturbation_scale_xy = 0.005 
    perturbation_scale_r = 0.001 
    
    x0_perturbed = np.copy(x0_base)
    x0_perturbed[0:2*n_circles] += rng.uniform(-perturbation_scale_xy, perturbation_scale_xy, 2*n_circles) 
    x0_perturbed[2*n_circles:3*n_circles] += rng.uniform(-perturbation_scale_r, perturbation_scale_r, n_circles) 
    
    # Ensure radii remain positive and within bounds (0.5 max) after perturbation
    x0_perturbed[2*n_circles:3*n_circles] = np.clip(x0_perturbed[2*n_circles:3*n_circles], MIN_RADIUS_EPSILON, MAX_POSSIBLE_RADIUS)
    # Ensure x,y coordinates remain within [0,1] after perturbation (before boundary constraints)
    x0_perturbed[0:2*n_circles] = np.clip(x0_perturbed[0:2*n_circles], 0.0, 1.0)
    
    return x0_perturbed

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES # Use the constant N_CIRCLES
    np.random.seed(42) # Set global seed for full determinism

    # --- Bounds for parameters ---
    lower_bounds = np.concatenate((np.full(n, 0.0), np.full(n, 0.0), np.full(n, MIN_RADIUS_EPSILON)))
    upper_bounds = np.concatenate((np.full(n, 1.0), np.full(n, 1.0), np.full(n, MAX_POSSIBLE_RADIUS)))
    bounds_scipy = Bounds(lower_bounds, upper_bounds)
    bounds_de = list(zip(lower_bounds, upper_bounds))

    # --- Constraints for scipy.optimize ---
    num_total_constraints = 4 * n + n * (n - 1) // 2
    nlc = NonlinearConstraint(lambda p: _evaluate_constraints_numba(p, n), 0, np.inf)
    slsqp_constraints = [{'type': 'ineq', 'fun': _evaluate_constraints_numba, 'args': (n,)}]

    # --- Initial Population for Differential Evolution (Hybrid Strategy) ---
    de_popsize = 60 # Increased popsize for better exploration given the strong initial guesses (from Insp 3, adjusted)
    initial_population = np.zeros((de_popsize, 3 * n))

    # Seed the population with good guesses (inspired by Inspiration 1)
    initial_population[0] = _generate_initial_grid_guess(n, 6, 6, random_perturbation=0.01, seed=45) # 6x6 grid
    initial_population[1] = _generate_initial_grid_guess(n, 5, 7, random_perturbation=0.02, seed=46) # 5x7 grid
    initial_population[2] = _generate_initial_grid_guess(n, 4, 8, random_perturbation=0.03, seed=47) # 4x8 grid
    initial_population[3] = _generate_hexagonal_guess(n) # Specific 32-circle hexagonal guess (from Insp 1)

    # Fill the rest with Latin Hypercube samples for diversity
    sampler = qmc.LatinHypercube(d=3 * n, seed=43)
    lhs_samples = sampler.random(n=de_popsize - 4)
    
    # Scale LHS samples to the problem bounds
    scaled_lhs = qmc.scale(lhs_samples, lower_bounds, upper_bounds)
    # Adjust initial radii for LHS to be slightly larger, but still valid (inspired by Insp 3)
    scaled_lhs[:, 2*n:3*n] = np.clip(scaled_lhs[:, 2*n:3*n], 0.01, MAX_POSSIBLE_RADIUS)
    initial_population[4:] = scaled_lhs

    print("Starting global optimization with Differential Evolution...")
    de_result = differential_evolution(
        _objective,
        bounds=bounds_de,
        args=(n,),
        constraints=(nlc,),
        strategy='best1bin',
        maxiter=3500,        # Increased iterations slightly from 2500 for more thorough search
        popsize=de_popsize,
        tol=1e-6,
        seed=42,
        disp=False,
        polish=False,
        workers=-1,
        init=initial_population
    )
    print(f"Differential Evolution finished. Best objective: {de_result.fun} (Sum Radii: {-de_result.fun:.4f})")
    
    x0_slsqp = de_result.x
    
    print("Starting local refinement with SLSQP...")
    slsqp_options = {'maxiter': 12000, 'ftol': 1e-9, 'eps': 1e-8, 'disp': False} # Slightly increased maxiter (from 10000)

    result = minimize(
        _objective,
        x0_slsqp,
        args=(n,),
        method='SLSQP',
        bounds=bounds_scipy,
        constraints=slsqp_constraints,
        options=slsqp_options
    )
    print(f"SLSQP finished. Final objective: {result.fun} (Sum Radii: {-result.fun:.4f})")

    if not result.success:
        print(f"Optimization failed: {result.message}")
        optimized_params = result.x if result.x is not None else de_result.x
    else:
        optimized_params = result.x

    # Ensure final radii are positive and within bounds
    optimized_params[2*n:3*n] = np.clip(optimized_params[2*n:3*n], MIN_RADIUS_EPSILON, MAX_POSSIBLE_RADIUS)

    # Reshape the optimized parameters into the desired (N, 3) format
    circles = np.zeros((n, 3))
    circles[:, 0] = optimized_params[0:n]
    circles[:, 1] = optimized_params[n:2*n]
    circles[:, 2] = optimized_params[2*n:3*n]

    # Final validation step: clip circle centers to be strictly within bounds based on their radii
    radii_final = circles[:, 2]
    circles[:, 0] = np.clip(circles[:, 0], radii_final, 1 - radii_final)
    circles[:, 1] = np.clip(circles[:, 1], radii_final, 1 - radii_final)

    return circles

# EVOLVE-BLOCK-END
