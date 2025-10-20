# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
from functools import partial
import time
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
import numba

# Constants for indexing flattened parameters
X_IDX = 0
Y_IDX = 1
R_IDX = 2

@numba.jit(nopython=True, fastmath=True)
def _jit_vectorized_constraints_fun(params_1d: np.ndarray, n_circles: int) -> np.ndarray:
    """
    Computes all constraint values as a single numpy array. Numba-optimized for performance.
    Each value must be >= 0 for the constraint to be satisfied.
    """
    circles = params_1d.reshape((n_circles, 3))
    x = circles[:, X_IDX]
    y = circles[:, Y_IDX]
    r = circles[:, R_IDX]

    # 1. Radii are non-negative (r_i >= 0)
    c_radius = r

    # 2. Containment within the unit square
    c_contain_x_min = x - r
    c_contain_x_max = 1.0 - x - r
    c_contain_y_min = y - r
    c_contain_y_max = 1.0 - y - r

    # 3. Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    # Re-implementing pairwise squared Euclidean distance and radii sums for Numba compatibility
    num_pairs = n_circles * (n_circles - 1) // 2
    c_overlap = np.empty(num_pairs, dtype=params_1d.dtype)
    pair_idx = 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy

            radii_sum = r[i] + r[j]
            min_dist_sq = radii_sum*radii_sum
            
            c_overlap[pair_idx] = dist_sq - min_dist_sq
            pair_idx += 1
    
    # Concatenate all constraints into a single vector
    all_constraints = np.concatenate((
        c_radius,
        c_contain_x_min,
        c_contain_x_max,
        c_contain_y_min,
        c_contain_y_max,
        c_overlap
    ))
    return all_constraints

@numba.jit(nopython=True, fastmath=True)
def _jit_vectorized_constraints_jac(params_1d: np.ndarray, n_circles: int) -> np.ndarray:
    """
    Computes the Jacobian matrix for all constraints. Numba-optimized for performance.
    Returns a 2D array (M x N_PARAMS).
    """
    n_params = n_circles * 3
    num_containment_constraints = 5 * n_circles
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    total_constraints = num_containment_constraints + num_overlap_constraints

    jac = np.zeros((total_constraints, n_params))
    circles = params_1d.reshape((n_circles, 3))

    # Containment constraints Jacobians
    # c_radius: r_k >= 0 (d/dr_k = 1)
    for k in range(n_circles):
        jac[k, k*3 + R_IDX] = 1.0
    
    # c_contain_x_min: x_k - r_k >= 0 (d/dx_k = 1, d/dr_k = -1)
    offset = n_circles
    for k in range(n_circles):
        jac[offset + k, k*3 + X_IDX] = 1.0
        jac[offset + k, k*3 + R_IDX] = -1.0
        
    # c_contain_x_max: 1 - x_k - r_k >= 0 (d/dx_k = -1, d/dr_k = -1)
    offset += n_circles
    for k in range(n_circles):
        jac[offset + k, k*3 + X_IDX] = -1.0
        jac[offset + k, k*3 + R_IDX] = -1.0
        
    # c_contain_y_min: y_k - r_k >= 0 (d/dy_k = 1, d/dr_k = -1)
    offset += n_circles
    for k in range(n_circles):
        jac[offset + k, k*3 + Y_IDX] = 1.0
        jac[offset + k, k*3 + R_IDX] = -1.0
        
    # c_contain_y_max: 1 - y_k - r_k >= 0 (d/dy_k = -1, d/dr_k = -1)
    offset += n_circles
    for k in range(n_circles):
        jac[offset + k, k*3 + Y_IDX] = -1.0
        jac[offset + k, k*3 + R_IDX] = -1.0

    # Non-overlap constraints Jacobians
    offset += n_circles
    constraint_idx = 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            xi, yi, ri = circles[i, 0], circles[i, 1], circles[i, 2]
            xj, yj, rj = circles[j, 0], circles[j, 1], circles[j, 2]
            
            jac[offset + constraint_idx, i*3 + X_IDX] = 2 * (xi - xj)
            jac[offset + constraint_idx, i*3 + Y_IDX] = 2 * (yi - yj)
            jac[offset + constraint_idx, i*3 + R_IDX] = -2 * (ri + rj)
            jac[offset + constraint_idx, j*3 + X_IDX] = -2 * (xi - xj)
            jac[offset + constraint_idx, j*3 + Y_IDX] = -2 * (yi - yj)
            jac[offset + constraint_idx, j*3 + R_IDX] = -2 * (ri + rj)
            constraint_idx += 1
    return jac

# Custom step function for basinhopping to perturb the solution
class MyTakeStep:
    def __init__(self, n_circles, bounds, stepsize_xy=0.05, stepsize_r=0.005):
        self.n = n_circles
        self.bounds_min = np.array([b[0] for b in bounds])
        self.bounds_max = np.array([b[1] for b in bounds])
        self.stepsize_xy = stepsize_xy
        self.stepsize_r = stepsize_r

    def __call__(self, x):
        x[X_IDX::3] += np.random.uniform(-self.stepsize_xy, self.stepsize_xy, self.n)
        x[Y_IDX::3] += np.random.uniform(-self.stepsize_xy, self.stepsize_xy, self.n)
        x[R_IDX::3] += np.random.uniform(-self.stepsize_r, self.stepsize_r, self.n)
        x = np.clip(x, self.bounds_min, self.bounds_max)
        return x

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses scipy.optimize.minimize with an SLSQP solver,
    starting from an initial grid-based packing, enhanced with basinhopping for global search
    and JIT-compiled constraint functions and Jacobians for high performance.
    """
    n = 32
    
    # 1. Initial Guess Generation
    grid_size = 6
    initial_radius = 1.0 / (2 * grid_size)
    initial_circles = np.zeros((n, 3))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n:
                initial_circles[count] = [(2 * i + 1) * initial_radius, (2 * j + 1) * initial_radius, initial_radius]
                count += 1
    initial_params_flat = initial_circles.flatten()

    # 2. Objective Function and Gradient
    def objective(params):
        return -np.sum(params[R_IDX::3])

    def gradient_objective(params, n_circles):
        grad = np.zeros_like(params)
        grad[R_IDX::3] = -1.0
        return grad

    # 3. Vectorized Constraint Functions and JIT-compiled Jacobian
    constraints_scipy_dict = {
        'type': 'ineq',
        'fun': partial(_jit_vectorized_constraints_fun, n_circles=n), # Use JIT-compiled constraint function
        'jac': partial(_jit_vectorized_constraints_jac, n_circles=n)
    }

    # 4. Bounds
    epsilon = 1e-7
    bounds = []
    for _ in range(n):
        bounds.extend([(epsilon, 1.0 - epsilon), (epsilon, 1.0 - epsilon), (epsilon, 0.5 - epsilon)])

    # 5. Helper function for parallel execution
    def _run_basinhopping_single_seed(seed, n_circles, objective, gradient_objective, constraints_dict, bounds, initial_params_flat, minimizer_options, niter, T, take_step_instance, initial_perturb_range):
        np.random.seed(seed) 
        perturbed_initial_params = initial_params_flat + np.random.uniform(-initial_perturb_range, initial_perturb_range, len(initial_params_flat))
        bounds_min = np.array([b[0] for b in bounds])
        bounds_max = np.array([b[1] for b in bounds])
        perturbed_initial_params = np.clip(perturbed_initial_params, bounds_min, bounds_max)

        minimizer_kwargs = {
            "method": "SLSQP", "bounds": bounds, "constraints": constraints_dict,
            "options": minimizer_options, "jac": partial(gradient_objective, n_circles=n_circles)
        }
        bh_result = basinhopping(
            func=objective, x0=perturbed_initial_params, minimizer_kwargs=minimizer_kwargs,
            niter=niter, T=T, take_step=take_step_instance, seed=seed, disp=False
        )
        return -objective(bh_result.x) if bh_result.x is not None else -np.inf, bh_result.x

    # 6. Perform Optimization
    minimizer_options = {'ftol': 1e-8, 'disp': False, 'maxiter': 3000} # Increased maxiter
    niter_per_run = 400 # Increased iterations
    T = 1.0
    initial_perturb_range = 0.01
    num_parallel_runs = 8 # Increased parallel runs
    base_seed = 42
    seeds = [base_seed + i for i in range(num_parallel_runs)]

    print(f"Starting {num_parallel_runs} parallel basinhopping runs, each with {niter_per_run} iterations...")
    take_step_instance = MyTakeStep(n_circles=n, bounds=bounds, stepsize_xy=0.05, stepsize_r=0.005)
    results = Parallel(n_jobs=-1)(
        delayed(_run_basinhopping_single_seed)(
            seed, n, objective, gradient_objective, constraints_scipy_dict, bounds,
            initial_params_flat, minimizer_options, niter_per_run, T,
            take_step_instance, initial_perturb_range
        ) for seed in seeds
    )
    print("Parallel basinhopping runs finished.")

    best_sum_radii, best_circles_params = max(results, key=lambda item: item[0], default=(-np.inf, None))

    # 7. Post-processing
    if best_circles_params is None:
        print("Warning: All parallel optimization runs failed. Returning initial guess.")
        optimized_params = initial_params_flat
    else:
        optimized_params = best_circles_params
        print(f"Overall best sum_radii found = {best_sum_radii:.6f}")

    circles = optimized_params.reshape((n, 3))
    circles[:, 2] = np.maximum(circles[:, 2], epsilon)
    return circles

# EVOLVE-BLOCK-END
