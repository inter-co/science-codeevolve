# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import jit
import time

# Define the number of circles globally for clarity and potential scalability
_N_CIRCLES = 26 

@jit(nopython=True)
def _unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpacks a 1D array of parameters [x1,y1,r1,x2,y2,r2,...] into (x, y, r) arrays.
    Using slicing for efficiency and Numba compatibility.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    return x, y, r

@jit(nopython=True)
def _objective(params: np.ndarray) -> float:
    """
    Objective function to minimize: negative sum of radii.
    Maximizing sum of radii is equivalent to minimizing its negative.
    """
    _, _, r = _unpack_params(params)
    return -np.sum(r)

@jit(nopython=True)
def _constraints_boundary(params: np.ndarray) -> np.ndarray:
    """
    Vectorized boundary constraints.
    Ensures r_i <= x_i <= 1-r_i and r_i <= y_i <= 1-r_i for all i.
    Returns an array where every element must be >= 0.
    """
    x, y, r = _unpack_params(params)
    n = len(r)
    # Constraints: x-r, 1-x-r, y-r, 1-y-r
    cons = np.empty(4 * n)
    cons[0::4] = x - r
    cons[1::4] = 1 - x - r
    cons[2::4] = y - r
    cons[3::4] = 1 - y - r
    return cons

@jit(nopython=True)
def _constraints_overlap(params: np.ndarray) -> np.ndarray:
    """
    Vectorized non-overlap constraints.
    Ensures sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj for all pairs i!=j.
    This is reformulated as (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0 to avoid sqrt.
    Returns an array where every element must be >= 0.
    """
    x, y, r = _unpack_params(params)
    n = len(r)
    num_pairs = n * (n - 1) // 2
    cons = np.empty(num_pairs)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            cons[k] = dist_sq - min_dist_sq
            k += 1
    return cons

def _generate_hexagonal_initial_guess(n_circles: int, seed: int) -> np.ndarray:
    """
    Generates a structured initial guess based on a perturbed hexagonal lattice.
    This provides a much better starting point than random placement.
    """
    np.random.seed(seed)
    
    # Configuration for 26 circles, known to form a good near-hexagonal packing
    rows_config = [5, 6, 5, 6, 4]
    if sum(rows_config) != n_circles:
        # Fallback for different N, though this is tailored for 26
        rows_config = [int(np.sqrt(n_circles))] * int(np.sqrt(n_circles))
    
    # Heuristic for radius based on the densest row
    max_n_cols = max(rows_config)
    # A tight packing in one row would have radius 1/(2*N). Leave a small margin.
    # Increased margin to 0.98 (from 0.95) to give slightly larger initial radii,
    # expecting the subsequent 0.98 shrinkage to still ensure feasibility.
    r = 1.0 / (2.0 * max_n_cols) * 0.98

    points = []
    y = r
    for i, n_cols in enumerate(rows_config):
        x = r
        if i % 2 == 1:
            x += r # Stagger odd rows for hexagonal packing
        for j in range(n_cols):
            if len(points) < n_circles:
                points.append([x, y, r])
                x += 2*r
        y += np.sqrt(3) * r

    initial_circles = np.array(points)

    # Center the entire generated pattern within the [0,1]x[0,1] square
    x_coords = initial_circles[:, 0]
    y_coords = initial_circles[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Bounding box of circles (not just centers)
    width = (x_max - x_min) + 2*r
    height = (y_max - y_min) + 2*r

    # Shift to center the bounding box
    x_shift = (1.0 - width) / 2.0 - (x_min - r)
    y_shift = (1.0 - height) / 2.0 - (y_min - r)
    
    initial_circles[:, 0] += x_shift
    initial_circles[:, 1] += y_shift

    # Add small random noise to break perfect symmetry, helping optimizer
    noise = np.random.normal(0, r * 0.05, size=initial_circles.shape)
    # Add small random noise to break perfect symmetry, helping optimizer
    # Noise for x, y coordinates
    xy_noise = np.random.normal(0, r * 0.05, size=(n_circles, 2))
    initial_circles[:, 0:2] += xy_noise
    
    # Noise for radii, allowing initial radii to vary slightly
    r_noise = np.random.normal(0, r * 0.02, size=n_circles) # 2% noise on radius
    initial_circles[:, 2] += r_noise
    
    # Ensure radii remain positive after noise
    initial_circles[:, 2] = np.maximum(initial_circles[:, 2], 1e-6)
    
    # Shrink radii slightly to ensure the initial guess is strictly feasible (no overlaps).
    # This gives the optimizer a "safe" starting point to expand from.
    # The value 0.98 makes the effective initial radius 0.98 * 0.98 = 0.9604 of the theoretical max for the lattice.
    initial_circles[:, 2] *= 0.98

    # Ensure initial guess is valid after shifting and adding noise, using the new smaller radii
    r_vals = initial_circles[:, 2]
    initial_circles[:, 0] = np.clip(initial_circles[:, 0], r_vals, 1 - r_vals)
    initial_circles[:, 1] = np.clip(initial_circles[:, 1], r_vals, 1 - r_vals)

    return initial_circles.flatten()

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    This version uses a single, deep optimization run starting from a structured
    (perturbed hexagonal) initial guess, with efficient vectorized constraints.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    # Define bounds for (x, y, r) for each circle parameter.
    # x, y in [0,1], radius in [epsilon, 0.5]
    all_bounds = []
    for _ in range(_N_CIRCLES):
        all_bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

    # Define constraints using the efficient vectorized functions.
    # This is significantly faster than using individual lambda functions.
    constraints = [
        {'type': 'ineq', 'fun': _constraints_boundary},
        {'type': 'ineq', 'fun': _constraints_overlap}
    ]

    # Try multiple optimization runs with slightly perturbed initial guesses
    # and a two-stage approach: a coarser initial run, then a finer one.
    
    best_sum_radii = 0.0
    best_circles_overall = np.zeros((_N_CIRCLES, 3))
    
    # Define optimization parameters for the fine-tuning stage
    slsqp_options_fine = {
        'disp': False, 
        'ftol': 1e-9, 
        'maxiter': 5000,
        'eps': 1e-7 # Step size for numerical approximation of gradients, adjusted for potentially better exploration
    }
    
    # Define optimization parameters for the coarse stage
    slsqp_options_coarse = {
        'disp': False, 
        'ftol': 1e-7, 
        'maxiter': 1000,
        'eps': 1e-6
    }
    
    # Validation tolerance for post-processing. Should be slightly looser than ftol or eps.
    validation_tolerance = 1e-7 # Loosen from 1e-9 to 1e-7 for robustness

    # Number of initial guesses to try (increased for better exploration)
    num_initial_guesses = 20 # Increased from 15 to 20 for broader exploration

    for i in range(num_initial_guesses):
        # Generate a high-quality initial guess based on a hexagonal lattice, with varying seeds for noise.
        # Use a different seed for each run to introduce variety in the initial perturbation.
        initial_params = _generate_hexagonal_initial_guess(_N_CIRCLES, seed=42 + i)
        
        # --- Stage 1: Coarser optimization with slightly relaxed parameters ---
        # This helps quickly escape poor local minima
        res_stage1 = minimize(
            _objective,
            initial_params,
            method='SLSQP',
            bounds=all_bounds,
            constraints=constraints,
            options=slsqp_options_coarse
        )

        # Use the refined parameters from stage 1 as the initial guess for stage 2,
        # or fall back to the original initial_params if stage 1 failed.
        current_params_for_stage2 = res_stage1.x if res_stage1.success else initial_params 
        
        # --- Stage 2: Finer optimization using the result from Stage 1 as initial guess ---
        res = minimize(
            _objective,
            current_params_for_stage2, # Use the refined parameters from stage 1
            method='SLSQP',
            bounds=all_bounds,
            constraints=constraints,
            options=slsqp_options_fine
        )

        current_circles = np.zeros((_N_CIRCLES, 3))
        if res.success:
            x_opt, y_opt, r_opt = _unpack_params(res.x)
            
            # Post-optimization validation to ensure the solution is physically valid.
            is_valid = True
            # Check boundary constraints with tolerance for floating point errors
            # (r_opt >= 0 - validation_tolerance) allows radii to be marginally negative due to floating point.
            # (r_opt <= x_opt + validation_tolerance) allows x to be marginally less than r.
            if not np.all((r_opt >= 0 - validation_tolerance) & 
                          (r_opt <= x_opt + validation_tolerance) & 
                          (x_opt <= 1 - r_opt + validation_tolerance) & 
                          (r_opt <= y_opt + validation_tolerance) & 
                          (y_opt <= 1 - r_opt + validation_tolerance)):
                is_valid = False
                # print(f"Warning (Run {i+1}): Solution violates boundary constraints.") # Commented for cleaner output
                
            # Check overlap constraints with tolerance for floating point errors
            if is_valid:
                for idx_i in range(_N_CIRCLES):
                    for idx_j in range(idx_i + 1, _N_CIRCLES):
                        dist_sq = (x_opt[idx_i] - x_opt[idx_j])**2 + (y_opt[idx_i] - y_opt[idx_j])**2
                        min_dist_sq = (r_opt[idx_i] + r_opt[idx_j])**2
                        # If dist_sq is significantly smaller than min_dist_sq, there's an overlap.
                        if dist_sq < min_dist_sq - validation_tolerance:
                            is_valid = False
                            # print(f"Warning (Run {i+1}): Circles {idx_i} and {idx_j} overlap.") # Commented for cleaner output
                            break
                    if not is_valid:
                        break
            
            if is_valid:
                current_circles = np.column_stack((x_opt, y_opt, r_opt))
                current_sum_radii = np.sum(current_circles[:, 2])
                
                if current_sum_radii > best_sum_radii:
                    best_sum_radii = current_sum_radii
                    best_circles_overall = current_circles
            else:
                # print(f"Warning (Run {i+1}): Optimizer reported success, but solution failed validation.") # Commented for cleaner output
                pass # Continue to next run
        else:
            # print(f"Warning (Run {i+1}): Optimization failed. Status: {res.status}, Message: {res.message}") # Commented for cleaner output
            pass # Continue to next run

    # Final check: if no valid solution with positive sum of radii was found across all runs
    if np.sum(best_circles_overall[:, 2]) <= 1e-5:
        print("Warning: No valid packing with positive sum of radii found after multiple runs. Returning default (zero) circles.")
        return np.zeros((_N_CIRCLES, 3))
        
    return best_circles_overall

# EVOLVE-BLOCK-END