# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit

# Numba-optimized helper functions for performance.
# The parameter vector `p` is structured as [x0,...,xn, y0,...,yn, r0,...,rn].
@njit
def array_to_pack(p, n):
    """Unpacks a 1D array into x, y, r coordinate arrays."""
    x = p[0:n]
    y = p[n:2*n]
    r = p[2*n:3*n]
    return x, y, r

@njit
def compute_penalty(p, n):
    """
    Computes a penalty score for a given circle configuration.
    The penalty is high for overlapping circles or circles outside the boundary.
    This function is the core of the physics-based relaxation.
    """
    x, y, r = array_to_pack(p, n)
    penalty = 0.0
    
    # Boundary constraints penalty: (x_i, y_i) must be at least r_i from the edge
    # The use of maximum(0, ...)^2 creates a smooth penalty function.
    penalty += np.sum(np.maximum(0, r - x)**2)
    penalty += np.sum(np.maximum(0, x + r - 1)**2)
    penalty += np.sum(np.maximum(0, r - y)**2)
    penalty += np.sum(np.maximum(0, y + r - 1)**2)

    # Pairwise overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            # Using squared distances to avoid costly sqrt operations
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            sum_radii = r[i] + r[j]
            overlap_sq_dist = sum_radii**2 - dist_sq
            # Add a penalty if circles overlap
            penalty += np.maximum(0, overlap_sq_dist)

    return penalty

def objective(p, n):
    """
    The main objective function to be minimized for SLSQP.
    We want to maximize the sum of radii, which is equivalent to minimizing
    the negative sum of radii. Constraints are handled explicitly by SLSQP.
    """
    radii = p[2*n:3*n]
    return -np.sum(radii)

# Numba-optimized constraint functions for SLSQP
@njit
def _boundary_constraints_numba(p, n):
    x, y, r = array_to_pack(p, n)
    # Each circle has 4 boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    constraints = np.empty(4 * n, dtype=p.dtype)
    for i in range(n):
        constraints[4*i + 0] = x[i] - r[i]       # x_i - r_i >= 0
        constraints[4*i + 1] = 1 - x[i] - r[i]   # 1 - x_i - r_i >= 0
        constraints[4*i + 2] = y[i] - r[i]       # y_i - r_i >= 0
        constraints[4*i + 3] = 1 - y[i] - r[i]   # 1 - y_i - r_i >= 0
    return constraints

@njit
def _overlap_constraints_numba(p, n):
    x, y, r = array_to_pack(p, n)
    num_overlap_constraints = n * (n - 1) // 2
    constraints = np.empty(num_overlap_constraints, dtype=p.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            sum_radii = r[i] + r[j]
            constraints[k] = dist_sq - sum_radii**2 # d^2 - (r_i+r_j)^2 >= 0
            k += 1
    return constraints

# Wrappers for scipy.optimize.minimize to call numba-compiled functions
# These are necessary because scipy.optimize.minimize expects functions,
# and passing the @njit function directly might not work as expected or lose jit benefits for the wrapper call itself.
# The internal computation within these wrappers will be jitted.
def boundary_constraints_wrapper(p, n):
    return _boundary_constraints_numba(p, n)

def overlap_constraints_wrapper(p, n):
    return _overlap_constraints_numba(p, n)

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses a two-stage optimization process:
    1. Initial Relaxation: Place circles of a fixed small radius and optimize their positions
       to minimize overlaps and boundary violations. This provides a good starting layout.
    2. Full Optimization: Optimize both positions and radii simultaneously to maximize the
       sum of radii, starting from the relaxed layout.
    
    This version performs multiple runs with different initial random seeds to overcome
    local optima issues and returns the best configuration found.
    """
    n = 26
    
    best_sum_radii = -np.inf
    best_circles = None
    
    num_runs = 50 # Significantly increased number of optimization attempts for broader exploration

    def generate_initial_positions(n_circles, rng):
        """Generates a structured initial layout (grid + random + perturbation)."""
        # For N=26, a 5x5 grid (25 circles) plus one random circle is a good start.
        num_grid_circles = min(n_circles, 25) 
        
        grid_x_coords = np.linspace(0.1, 0.9, 5)
        grid_y_coords = np.linspace(0.1, 0.9, 5)
        
        xx, yy = np.meshgrid(grid_x_coords, grid_y_coords)
        
        x_coords = xx.flatten()[:num_grid_circles]
        y_coords = yy.flatten()[:num_grid_circles]
        
        if n_circles > num_grid_circles:
            # Add remaining circles randomly
            remaining_circles = n_circles - num_grid_circles
            x_coords = np.concatenate((x_coords, rng.uniform(0.1, 0.9, remaining_circles)))
            y_coords = np.concatenate((y_coords, rng.uniform(0.1, 0.9, remaining_circles)))
                
        # Add a slightly larger random perturbation to all positions for diversity
        perturbation_scale = 0.03 
        x_coords += rng.uniform(-perturbation_scale, perturbation_scale, n_circles)
        y_coords += rng.uniform(-perturbation_scale, perturbation_scale, n_circles)
        
        # Clip to ensure positions stay within reasonable initial bounds
        x_coords = np.clip(x_coords, 0.05, 0.95)
        y_coords = np.clip(y_coords, 0.05, 0.95)
        
        return x_coords, y_coords
    
    for run_idx in range(num_runs):
        seed = 42 + run_idx # Vary the seed for each run to get different initial configurations
        rng = np.random.default_rng(seed)

        # --- Stage 1: Initial Guess and Relaxation ---
        # Start with a structured configuration for better initial spread.
        initial_x, initial_y = generate_initial_positions(n, rng)
        # Vary initial radii slightly for relaxation with a slightly tighter range to introduce more diversity
        initial_r_relax = rng.uniform(0.035, 0.065, n) 
        
        p0_relax = np.concatenate((initial_x, initial_y, initial_r_relax))
        
        # The relaxation objective only minimizes the penalty to spread circles apart.
        def relaxation_objective(pos_vars, fixed_radii):
            p_relax = np.concatenate((pos_vars, fixed_radii))
            return compute_penalty(p_relax, n)

        pos_bounds = [(0.0, 1.0)] * (2 * n)
        
        # Run a short optimization to find a good initial placement.
        res_relax = minimize(
            fun=relaxation_objective,
            x0=p0_relax[:2*n],
            args=(p0_relax[2*n:],),
            method='L-BFGS-B',
            bounds=pos_bounds,
            options={'maxiter': 750, 'ftol': 1e-7} # maxiter for relaxation stage unchanged
        )
        
        # --- Stage 2: Full Optimization (Positions and Radii) ---
        # Use the relaxed positions as the starting point for the main optimization.
        # Initialize radii for Stage 2 by scaling up the relaxed radii more aggressively to encourage inflation.
        # This propagates the initial radius diversity from stage 1.
        initial_r_full = np.clip(initial_r_relax * 1.35, 0.01, 0.5) # More aggressive scaling
        p0_full = np.concatenate((res_relax.x, initial_r_full))
        
        # Bounds for the full optimization problem: x, y in [0,1], r in [0.01, 0.5]
        # A small minimum radius prevents circles from disappearing. Max radius increased to allow more flexibility.
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.5)] * n

        # Define constraints for SLSQP
        constraints = [
            {'type': 'ineq', 'fun': boundary_constraints_wrapper, 'args': (n,)},
            {'type': 'ineq', 'fun': overlap_constraints_wrapper, 'args': (n,)}
        ]

        # Run the main optimization to inflate radii and fine-tune positions using SLSQP.
        # The objective function no longer includes a penalty term, as constraints are handled explicitly.
        result = minimize(
            fun=objective, # The modified objective function (negative sum of radii)
            x0=p0_full,
            args=(n,), # Only 'n' is passed to the objective, no penalty_weight
            method='SLSQP', # Switched to SLSQP for explicit constraint handling
            bounds=bounds,
            constraints=constraints, # Added explicit constraints
            options={'maxiter': 7500, 'ftol': 1e-10, 'disp': False} # Increased maxiter for SLSQP
        )
        
        # Check if the optimization was successful and improved the result
        if result.success:
            current_radii = result.x[2*n:3*n]
            current_sum_radii = np.sum(current_radii)
            
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                final_x = result.x[0:n]
                final_y = result.x[n:2*n]
                final_r = result.x[2*n:3*n]
                best_circles = np.vstack((final_x, final_y, final_r)).T
        # else:
            # print(f"Warning: Optimization run {run_idx} failed: {result.message}")
    
    # Return the best configuration found across all runs
    if best_circles is None:
        # Fallback if all runs fail, or if num_runs is 0 (shouldn't happen with default num_runs > 0)
        raise RuntimeError("No successful circle packing configuration found after multiple runs.")
        
    return best_circles

# EVOLVE-BLOCK-END