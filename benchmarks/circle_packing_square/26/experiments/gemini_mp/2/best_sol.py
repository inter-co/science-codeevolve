# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping # Add basinhopping for global optimization
from scipy.spatial.distance import pdist # For efficient distance calculations
from joblib import Parallel, delayed # For parallelization of multiple trials

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26

    # Objective, bounds, and constraints are defined once, as they are static.

    # 2. Objective Function: Minimize negative sum of radii
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    # Gradient for the objective function
    def jac_objective(params):
        grad = np.zeros_like(params)
        grad[2::3] = -1.0 # d(-sum(r_i))/dr_i = -1
        return grad

    # 3. Bounds for x, y, r
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
        bounds.append((1e-6, 0.5)) # r_i (small lower bound to ensure positivity)

    # 4. Constraint Functions and their Jacobians
    _triu_indices = np.triu_indices(n, k=1) # Precompute for non-overlap

    # Containment constraints (4 per circle)
    def containment_constraints(params):
        x, y, r = params[0::3], params[1::3], params[2::3]
        c = np.empty(4 * n)
        c[0*n:1*n] = x - r         # x_i - r_i >= 0
        c[1*n:2*n] = 1 - x - r     # 1 - x_i - r_i >= 0
        c[2*n:3*n] = y - r         # y_i - r_i >= 0
        c[3*n:4*n] = 1 - y - r     # 1 - y_i - r_i >= 0
        return c

    # Jacobian for containment constraints
    def jac_containment_constraints(params):
        grad = np.zeros((4 * n, 3 * n))
        # d(x_i - r_i)/dx_i = 1, d(x_i - r_i)/dr_i = -1
        grad[np.arange(n), np.arange(n)*3] = 1.0
        grad[np.arange(n), np.arange(n)*3 + 2] = -1.0
        # d(1 - x_i - r_i)/dx_i = -1, d(1 - x_i - r_i)/dr_i = -1
        grad[n + np.arange(n), np.arange(n)*3] = -1.0
        grad[n + np.arange(n), np.arange(n)*3 + 2] = -1.0
        # d(y_i - r_i)/dy_i = 1, d(y_i - r_i)/dr_i = -1
        grad[2*n + np.arange(n), np.arange(n)*3 + 1] = 1.0
        grad[2*n + np.arange(n), np.arange(n)*3 + 2] = -1.0
        # d(1 - y_i - r_i)/dy_i = -1, d(1 - y_i - r_i)/dr_i = -1
        grad[3*n + np.arange(n), np.arange(n)*3 + 1] = -1.0
        grad[3*n + np.arange(n), np.arange(n)*3 + 2] = -1.0
        return grad

    # Non-overlap constraints (n*(n-1)/2 pairs)
    def non_overlap_constraints(params):
        x, y, r = params[0::3], params[1::3], params[2::3]
        coords = np.vstack((x, y)).T
        dist_sq = pdist(coords, 'sqeuclidean')
        r_i, r_j = r[_triu_indices[0]], r[_triu_indices[1]]
        min_dist_sq = (r_i + r_j)**2
        return dist_sq - min_dist_sq

    # Jacobian for non-overlap constraints (vectorized)
    def jac_non_overlap_constraints(params):
        x, y, r = params[0::3], params[1::3], params[2::3]
        
        i_indices, j_indices = _triu_indices[0], _triu_indices[1]
        
        x_diff = x[i_indices] - x[j_indices]
        y_diff = y[i_indices] - y[j_indices]
        r_sum = r[i_indices] + r[j_indices]
        
        num_pairs = len(i_indices)
        grad = np.zeros((num_pairs, 3 * n))
        
        # Derivatives w.r.t. x_i, y_i, r_i
        grad[np.arange(num_pairs), i_indices * 3] = 2 * x_diff
        grad[np.arange(num_pairs), i_indices * 3 + 1] = 2 * y_diff
        grad[np.arange(num_pairs), i_indices * 3 + 2] = -2 * r_sum
        
        # Derivatives w.r.t. x_j, y_j, r_j
        grad[np.arange(num_pairs), j_indices * 3] = -2 * x_diff
        grad[np.arange(num_pairs), j_indices * 3 + 1] = -2 * y_diff
        grad[np.arange(num_pairs), j_indices * 3 + 2] = -2 * r_sum
        
        return grad

    constraints = [
        {'type': 'ineq', 'fun': containment_constraints, 'jac': jac_containment_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints, 'jac': jac_non_overlap_constraints}
    ]

    # Helper function to run a single basinhopping trial with a specific seed
    def run_basinhopping_trial(trial_seed):
        np.random.seed(trial_seed) # Set seed for initial guess generation

        # Generate a new Initial Guess (x0) for each trial
        x0 = np.zeros(n * 3)
        grid_dim, num_grid_circles = 5, 25
        initial_r_grid, perturbation_strength = 0.05, 0.01
        
        idx = 0
        for i in range(grid_dim):
            for j in range(grid_dim):
                if idx < num_grid_circles:
                    x_center = (i + 0.5) / grid_dim
                    y_center = (j + 0.5) / grid_dim
                    x0[idx*3+0] = np.clip(x_center + np.random.uniform(-perturbation_strength, perturbation_strength), 0, 1)
                    x0[idx*3+1] = np.clip(y_center + np.random.uniform(-perturbation_strength, perturbation_strength), 0, 1)
                    x0[idx*3+2] = initial_r_grid
                    idx += 1

        for k in range(idx, n):
            x0[k*3+0] = np.random.uniform(0.1, 0.9)
            x0[k*3+1] = np.random.uniform(0.1, 0.9)
            x0[k*3+2] = initial_r_grid

        # Minimizer arguments for basinhopping's local optimization step (SLSQP)
        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": constraints,
            "jac": jac_objective, # Provide analytical Jacobian for the objective
            "options": {'maxiter': 1000, 'ftol': 1e-7, 'disp': False} # Reduced maxiter, slightly relaxed ftol for faster local convergence
        }

        # Perform basinhopping optimization
        # niter: Number of global hopping iterations
        # T: Temperature parameter for the Metropolis-Hastings criterion (controls acceptance probability)
        # stepsize: Maximum step size for random perturbations
        bh_result = basinhopping(objective, x0,
                                 minimizer_kwargs=minimizer_kwargs,
                                 niter=200, # Reduced niter to decrease overall computation per trial
                                 T=1.0,      # Default temperature, can be tuned
                                 stepsize=0.05, # Reasonable step size for perturbations
                                 disp=False, # Suppress intermediate output for cleaner parallel runs
                                 seed=trial_seed # For reproducibility of the basinhopping process itself
                                )
        return bh_result

    # Multi-start Optimization using basinhopping and parallelization
    n_trials = 8 # Increased number of independent basinhopping runs for broader exploration
    # Use joblib to run trials in parallel across available CPU cores.
    # 'loky' backend is generally more robust for multiprocessing.
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_basinhopping_trial)(42 + trial) for trial in range(n_trials)
    )

    best_result = None
    best_fun = np.inf

    for result in results:
        # basinhopping result object contains 'fun' (best objective value found) and
        # 'lowest_optimization_result' (the scipy.optimize.OptimizeResult object for that best point).
        # We check the success of the local optimization that found the lowest function value.
        if result.lowest_optimization_result.success and result.fun < best_fun:
            best_fun = result.fun
            best_result = result.lowest_optimization_result # Store the best local optimization result

    if best_result is None:
        print("All basinhopping trials failed or did not find a successful local minimum.")
        return np.zeros((n, 3))

    if not best_result.success:
        print(f"Best found solution was from a failed local optimization within basinhopping: {best_result.message}")

    # Reshape the best parameters found across all trials
    optimal_params = best_result.x
    circles = optimal_params.reshape(n, 3)
    
    # Ensure radii are non-negative, as optimization might yield very small negative values due to precision.
    circles[:, 2] = np.maximum(circles[:, 2], 0)

    # Post-optimization validation to rigorously check constraints
    if not validate_packing(circles):
        print("WARNING: Final packing failed validation checks! The solution might be infeasible.")
        # In a real-world scenario, one might log the failed constraints or attempt re-optimization.
    
    return circles

def validate_packing(circles: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validates if a given set of circles satisfies containment and non-overlap constraints.

    Args:
        circles: np.array of shape (N,3), where each row is (x,y,r).
        tolerance: Numerical tolerance for constraint checks (e.g., f(x) >= -tolerance).

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    n = circles.shape[0]
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # 1. Containment constraints: ri <= xi <= 1-ri and ri <= yi <= 1-ri
    # Which translates to: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    if not np.all(x - r >= -tolerance):
        return False
    if not np.all(1 - x - r >= -tolerance):
        return False
    if not np.all(y - r >= -tolerance):
        return False
    if not np.all(1 - y - r >= -tolerance):
        return False

    # 2. Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    coords = circles[:, :2]
    radii = circles[:, 2]
    for i in range(n):
        for j in range(i + 1, n): # Only check each pair once
            dist_centers = np.linalg.norm(coords[i] - coords[j])
            if dist_centers < radii[i] + radii[j] - tolerance:
                return False
    
    return True

# EVOLVE-BLOCK-END