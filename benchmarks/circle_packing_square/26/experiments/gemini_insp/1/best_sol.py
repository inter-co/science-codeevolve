# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, basinhopping # Added basinhopping
import numba

@numba.jit(nopython=True, fastmath=True)
def _non_overlap_constraints_numba(pos: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Calculates non-overlap constraints efficiently using Numba.
    Constraint is (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    """
    n = pos.shape[0]
    num_constraints = n * (n - 1) // 2
    out = np.empty(num_constraints, dtype=pos.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            sq_dist = dx*dx + dy*dy
            
            r_sum = radii[i] + radii[j]
            r_sum_sq = r_sum * r_sum
            
            out[k] = sq_dist - r_sum_sq
            k += 1
    return out

def circle_packing26() -> np.ndarray:
    """
    Finds an optimal arrangement of 26 non-overlapping circles in a unit square
    to maximize the sum of their radii using a hybrid approach:
    Numba-JIT compiled constraints for speed, and basinhopping for global search
    to escape local minima, with SLSQP as the local optimizer.
    """
    n = 26
    # For reproducibility of the initial guess and basinhopping's random steps
    np.random.seed(42)

    # The state vector X is a flat array: [x0, y0, r0, x1, y1, r1, ...]
    # The objective is to maximize sum of radii, which is equivalent to minimizing -sum(radii).
    def objective(X):
        radii = X[2::3]
        return -np.sum(radii)

    # All constraints are formulated as g(X) >= 0.
    def constraints(X):
        circles = X.reshape((n, 3))
        pos = circles[:, :2]
        radii = circles[:, 2]

        # Constraint 1: Containment within the unit square [0,1] x [0,1]
        containment_constraints = np.concatenate([
            pos[:, 0] - radii,         # x - r >= 0
            1.0 - pos[:, 0] - radii,   # 1 - x - r >= 0
            pos[:, 1] - radii,         # y - r >= 0
            1.0 - pos[:, 1] - radii    # 1 - y - r >= 0
        ])

        # Constraint 2: Non-overlapping circles (calculated with Numba)
        if n > 1:
            non_overlap_constraints = _non_overlap_constraints_numba(pos, radii)
        else:
            non_overlap_constraints = np.array([])

        return np.concatenate([containment_constraints, non_overlap_constraints])

    # Initial Guess (x0):
    # A feasible and reasonably distributed starting point on a 5x6 grid.
    initial_circles = np.zeros((n, 3))
    num_rows, num_cols = 5, 6
    r_base = 1.0 / (2.0 * max(num_rows, num_cols)) * 0.95
    if r_base < 1e-6: r_base = 1e-6
    
    x_coords = np.linspace(r_base, 1.0 - r_base, num_cols)
    y_coords = np.linspace(r_base, 1.0 - r_base, num_rows)
    
    idx = 0
    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            if idx < n:
                initial_circles[idx, 0] = x_coords[c_idx]
                initial_circles[idx, 1] = y_coords[r_idx]
                initial_circles[idx, 2] = r_base
                idx += 1
    
    # Add small random perturbation to break symmetry and aid exploration.
    perturb_scale = 0.005
    initial_circles[:, :2] += np.random.uniform(-perturb_scale, perturb_scale, (n, 2))
    initial_circles[:, 2] += np.random.uniform(-r_base * 0.05, r_base * 0.05, n)
    
    initial_circles[:, 2] = np.clip(initial_circles[:, 2], 1e-6, 0.5)
    
    # Re-enforce containment after perturbation to ensure initial feasibility.
    for i in range(n):
        r_i = initial_circles[i, 2]
        initial_circles[i, 0] = np.clip(initial_circles[i, 0], r_i, 1.0 - r_i)
        initial_circles[i, 1] = np.clip(initial_circles[i, 1], r_i, 1.0 - r_i)

    x0 = initial_circles.ravel()

    # Define bounds for each variable: 0<=x,y<=1 and 0<=r<=0.5
    bounds = Bounds([0.0, 0.0, 1e-6] * n, [1.0, 1.0, 0.5] * n)

    # Define the constraints for the optimizer
    cons = {'type': 'ineq', 'fun': constraints}

    # Options for the local minimizer (SLSQP) within basinhopping
    minimizer_options = {'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
    
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": cons,
        "options": minimizer_options
    }

    # Run global optimization using basinhopping
    # niter: number of hopping iterations
    # T: temperature for the Metropolis-Hastings acceptance criterion
    # stepsize: maximum step size for random perturbations
    result = basinhopping(objective,
                          x0,
                          minimizer_kwargs=minimizer_kwargs,
                          niter=100, # Number of global search iterations
                          T=1.0,     # Temperature for acceptance criteria
                          stepsize=0.05, # Max step for random perturbation
                          seed=42,   # For reproducibility of basinhopping's random steps
                          disp=False) # Set to True for verbose output

    # basinhopping returns the best solution found across all local minimizations
    final_circles = result.x.reshape((n, 3))

    # A final check to ensure the solution is reasonably feasible
    # basinhopping itself does not guarantee constraint satisfaction if local minimizer fails.
    final_constraint_values = constraints(result.x)
    # Check if all constraints are met within a small tolerance.
    # Relaxed tolerance slightly for basinhopping results as it aims for global best, not strict local feasibility.
    if np.all(final_constraint_values >= -1e-5): 
        return final_circles
    else:
        print("Warning: Basinhopping found a solution, but constraints are significantly violated.")
        # If violations are significant (e.g., > -1e-5), one might consider an additional
        # SLSQP run from `result.x` with tighter tolerances or print more detailed error.
        # For this problem, we return the best found solution.
        return final_circles

# EVOLVE-BLOCK-END