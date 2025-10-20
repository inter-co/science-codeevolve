# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
from numba import njit, float64

# Constants for the problem (from Inspiration 3)
N_CIRCLES = 32
MIN_RADIUS_BOUND = 0.001 # Minimum radius to prevent degenerate circles and numerical issues
FIXED_SEED = 42 # For reproducibility of random elements

# Numba-jitted objective function (from Inspiration 3)
@njit(float64(float64[:]), cache=True)
def _objective_numba(params):
    """
    Objective function to minimize: negative sum of radii.
    params: 1D numpy array of [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    r = params[2::3] 
    return -np.sum(r)

# Numba-jitted constraint function (from Inspiration 3, including positive radius constraint)
@njit(float64[:](float64[:]), cache=True)
def _constraints_numba(params):
    """
    Evaluates all constraints (boundary and non-overlap) for a given set of circle parameters.
    Returns a 1D array of constraint values, all of which must be >= 0.
    
    params: 1D numpy array of [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    """
    n = N_CIRCLES # Use the global constant
    
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    num_boundary_constraints = 4 * n
    num_positive_radius_constraints = n # New: Constraint for minimum radius
    num_overlap_constraints = n * (n - 1) // 2
    
    total_constraints = num_boundary_constraints + num_positive_radius_constraints + num_overlap_constraints
    constraints_values = np.empty(total_constraints, dtype=np.float64)
    
    idx = 0

    # 1. Boundary constraints (4*n)
    # ri <= xi <= 1-ri  => xi - ri >= 0  AND  1 - xi - ri >= 0
    # ri <= yi <= 1-ri  => yi - ri >= 0  AND  1 - yi - ri >= 0
    for i in range(n):
        constraints_values[idx] = x[i] - r[i]
        idx += 1
        constraints_values[idx] = 1.0 - x[i] - r[i]
        idx += 1
        constraints_values[idx] = y[i] - r[i]
        idx += 1
        constraints_values[idx] = 1.0 - y[i] - r[i]
        idx += 1

    # 2. Positive radius constraints (n) (from Inspiration 2 and 3)
    # r >= MIN_RADIUS_BOUND (ensure radii are positive, preventing collapse to zero)
    for i in range(n):
        constraints_values[idx] = r[i] - MIN_RADIUS_BOUND
        idx += 1

    # 3. Non-overlap constraints (n*(n-1)/2)
    # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # So, (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist_sq = (r[i] + r[j]) * (r[i] + r[j])
            constraints_values[idx] = dist_sq - min_dist_sq
            idx += 1
            
    return constraints_values


def circle_packing32()->np.ndarray:
    """
    Finds an optimal arrangement of 32 non-overlapping circles in a unit square
    to maximize the sum of their radii using a constrained optimization approach.

    The problem is formulated for a hybrid Differential Evolution (global)
    and Sequential Least Squares Programming (local) optimizer. It starts from
    a jittered grid configuration and iteratively adjusts the circles' positions
    (x, y) and radii (r) to maximize the sum of radii while satisfying boundary
    and non-overlap constraints.

    Returns:
        np.ndarray: An array of shape (32, 3) where each row represents a
                    circle as [x_center, y_center, radius].
    """
    # Use global constant N_CIRCLES
    
    def _generate_initial_guess(seed: int) -> np.ndarray: # Removed n_circles arg, uses global N_CIRCLES
        """
        Generates an initial configuration for circle packing (adapted from target).
        Uses a 6x6 grid, samples N_CIRCLES points, jitters them, and assigns a
        bimodal distribution of initial radii to encourage size diversity.
        """
        rng = np.random.default_rng(seed)

        # Use a more square-like 6x6 grid and sample N_CIRCLES points to break symmetry.
        grid_dim = 6
        n_grid_points = grid_dim * grid_dim
        
        # Calculate a base radius for the grid.
        base_radius = 1 / (2 * grid_dim) # Approx 1/12 = 0.083
        
        # Generate grid coordinates
        x_coords = np.linspace(base_radius, 1 - base_radius, grid_dim)
        y_coords = np.linspace(base_radius, 1 - base_radius, grid_dim)
        xx, yy = np.meshgrid(x_coords, y_coords)
        all_centers = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Randomly select N_CIRCLES out of 36 grid points
        indices = rng.choice(n_grid_points, N_CIRCLES, replace=False)
        initial_centers = all_centers[indices]
        
        # Add jitter to further break symmetry, scaled by the base radius
        # Increased jitter magnitude for more initial diversity (from analysis)
        jitter_magnitude = base_radius * 0.5
        initial_centers += rng.uniform(-jitter_magnitude, jitter_magnitude, initial_centers.shape)
        
        # Ensure centers stay within bounds after jittering
        initial_centers = np.clip(initial_centers, 0.0, 1.0)
        
        # Introduce a bimodal distribution for initial radii to encourage size diversity.
        initial_radii = np.zeros(N_CIRCLES)
        num_large_circles = N_CIRCLES // 4  # Designate ~1/4 of circles to be potentially larger
        large_indices = rng.choice(N_CIRCLES, num_large_circles, replace=False)
        small_indices = np.setdiff1d(np.arange(N_CIRCLES), large_indices)

        # Larger circles can start up to 1.8x base_radius
        initial_radii[large_indices] = rng.uniform(base_radius * 0.8, base_radius * 1.8, num_large_circles)
        # Smaller circles fill the gaps
        initial_radii[small_indices] = rng.uniform(base_radius * 0.4, base_radius * 1.0, N_CIRCLES - num_large_circles)
        
        # Ensure initial radii are at least MIN_RADIUS_BOUND
        initial_radii = np.maximum(initial_radii, MIN_RADIUS_BOUND)

        # Flatten parameters into a 1D array for the optimizer: [x0,y0,r0, x1,y1,r1, ...].
        x0 = np.hstack([initial_centers, initial_radii.reshape(-1, 1)]).ravel()
        return x0

    # Multi-start optimization setup
    # Further increased restarts for broader search of the solution space (from analysis)
    num_restarts = 40
    
    best_circles = None
    best_sum_radii = -np.inf 
    
    # Store an initial guess for fallback if all restarts fail
    x0_fallback = _generate_initial_guess(FIXED_SEED) # Call without n_circles arg, uses global FIXED_SEED

    # Objective Function (to be minimized). Uses the Numba-jitted function directly.
    objective = _objective_numba

    # Bounds for variables [x, y, r] for each circle (from Inspiration 2/3).
    # A small positive lower bound on radius prevents degenerate cases.
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (MIN_RADIUS_BOUND, 0.5)])

    # Nonlinear Constraint definition (from Inspiration 2/3)
    # The _constraints_numba function returns an array of values `g_k(params)`.
    # `lb` (lower bound) for these values is 0, `ub` (upper bound) is infinity.
    num_total_constraints = 4 * N_CIRCLES + N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    nonlinear_constraint = NonlinearConstraint(
        _constraints_numba, 
        lb=np.zeros(num_total_constraints), 
        ub=np.full(num_total_constraints, np.inf)
    )

    # Run the optimizer multiple times with different initial guesses (Hybrid DE + SLSQP, from Inspiration 2/3).
    for i in range(num_restarts):
        current_seed = FIXED_SEED + i # Generate unique seed for each restart
        
        # --- Initial Guess Strategy for current restart ---
        x0_current_restart = _generate_initial_guess(current_seed)

        # --- Optimization using a hybrid approach: Differential Evolution + SLSQP ---
        # 1. Global Optimization with Differential Evolution (tuned from Inspiration 3 and further refined)
        de_options = {
            'maxiter': 10000,       # Significantly increased iterations for deeper global exploration
            'popsize': 40,          # Increased population multiplier for more diversity (from analysis)
            'disp': False,
            'workers': -1,          # Use all available CPU cores
            'polish': False,        # No local refinement after DE itself; SLSQP handles it
            'updating': 'deferred', # Better for parallel workers
            'tol': 1e-6,            # Tighten tolerance for DE convergence
        }
        
        try:
            de_result = differential_evolution(
                _objective_numba, # Use jitted objective
                bounds,
                constraints=[nonlinear_constraint], # Use the NonlinearConstraint object
                strategy='currenttobest1bin', # More aggressive strategy for faster convergence
                maxiter=de_options['maxiter'],
                popsize=de_options['popsize'],
                disp=de_options['disp'],
                workers=de_options['workers'],
                polish=de_options['polish'],
                updating=de_options['updating'],
                seed=current_seed, # Use current restart's seed for reproducibility
                # Reduced constraint penalty to allow DE more exploration, trusting SLSQP for final feasibility (from analysis)
                constraint_penalty=20.0
            )
            x0_refined = de_result.x # Use DE's best result as initial guess for SLSQP
            
        except Exception as e:
            # print(f"Restart {i+1}: Warning - Differential Evolution failed: {e}. Falling back to initial guess for SLSQP.")
            x0_refined = x0_current_restart # Fallback to initial guess if DE fails

        # 2. Local Refinement with SLSQP (tuned from Inspiration 3 and further refined)
        slsqp_options = {
            'maxiter': 15000, # Increased max iterations for better convergence after DE (from analysis)
            'ftol': 1e-11,    # Further tighten function tolerance for termination
            'disp': False,
            'eps': 1e-7,      # Adjust step size for finite difference approximation of gradients
        }
        
        try:
            result = minimize(
                _objective_numba, # Use jitted objective
                x0_refined, # Use the refined initial guess from Differential Evolution
                method='SLSQP',
                bounds=bounds,
                constraints=[nonlinear_constraint], # Use the NonlinearConstraint object
                options=slsqp_options
            )

            current_sum_radii = -result.fun

            if result.success and current_sum_radii > best_sum_radii:
                # Post-processing: ensure radii are strictly positive, correcting potential float precision issues.
                optimized_circles_current = result.x.reshape((N_CIRCLES, 3))
                optimized_circles_current[:, 2] = np.maximum(optimized_circles_current[:, 2], MIN_RADIUS_BOUND)
                
                # Re-evaluate constraints for the best result to ensure true feasibility
                final_constraint_violations = _constraints_numba(optimized_circles_current.ravel())
                # Allow a tiny tolerance for floating point errors, e.g., 1e-7
                if np.all(final_constraint_violations >= -1e-7): 
                    best_sum_radii = current_sum_radii
                    best_circles = optimized_circles_current
                # else:
                    # print(f"Restart {i+1}: Solution is locally better but has minor constraint violations.")

        except Exception as e:
            # print(f"Restart {i+1}: Warning - SLSQP optimization failed: {e}")
            pass # Continue to next restart

    # 6. Process and return the best result found across all restarts.
    if best_circles is not None:
        # Final check for constraint satisfaction on the best_circles
        final_params = best_circles.ravel()
        final_violations = _constraints_numba(final_params)
        # Use a slightly larger tolerance for printing warnings, as optimizers can sometimes be slightly off.
        if np.any(final_violations < -1e-6): 
            print(f"Warning: The best found solution has minor constraint violations, sum_radii: {best_sum_radii:.5f}. Max violation: {np.min(final_violations):.2e}")
        return best_circles
    else:
        print(f"Warning: All {num_restarts} optimizations failed. Returning the initial guess from base seed {FIXED_SEED}.")
        return x0_fallback.reshape((N_CIRCLES,3))

# EVOLVE-BLOCK-END
