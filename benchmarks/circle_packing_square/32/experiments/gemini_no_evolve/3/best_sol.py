# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import numba

# Using Numba to accelerate the O(n^2) overlap calculation, which is the bottleneck.
# The `cache=True` option saves the compiled function to disk for faster subsequent runs.
@numba.jit(nopython=True, cache=True)
def non_overlap_constraint_numba(coords, radii, n):
    """Calculates the non-overlap constraint for all unique pairs of circles."""
    num_pairs = n * (n - 1) // 2
    constraints = np.empty(num_pairs, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            dist_sq = (coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2
            radii_sum = radii[i] + radii[j]
            constraints[k] = dist_sq - radii_sum**2
            k += 1
    return constraints

def circle_packing32()->np.ndarray:
    """
    Finds an optimal arrangement of 32 non-overlapping circles in a unit square
    to maximize the sum of their radii, using the SLSQP optimization algorithm.
    """
    n = 32
    
    # The parameters for the optimizer are a flat array: [x0,y0,r0, x1,y1,r1, ...]
    # Total variables: 32 * 3 = 96

    # Objective function to minimize: the negative sum of radii.
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    # Constraint function: all values must be non-negative for a feasible solution.
    def constraints_func(params):
        circles = params.reshape((n, 3))
        coords = circles[:, :2]
        radii = circles[:, 2]

        # 1. Boundary constraints (4 * n constraints)
        # ri <= xi <= 1-ri  =>  xi - ri >= 0  and  1 - xi - ri >= 0
        # ri <= yi <= 1-ri  =>  yi - ri >= 0  and  1 - yi - ri >= 0
        boundary_constraints = np.concatenate([
            coords[:, 0] - radii,      # x_i - r_i >= 0
            1 - coords[:, 0] - radii,  # 1 - x_i - r_i >= 0
            coords[:, 1] - radii,      # y_i - r_i >= 0
            1 - coords[:, 1] - radii   # 1 - y_i - r_i >= 0
        ])

        # 2. Non-overlap constraints (n * (n-1) / 2 = 496 constraints)
        overlap_constraints = non_overlap_constraint_numba(coords, radii, n)
        
        return np.concatenate([boundary_constraints, overlap_constraints])

    # Initial guess (x0): a simple grid layout to start from a feasible state.
    # A 6x6 grid has 36 spots; we use the first 32.
    grid_size = 6
    spacing = 1.0 / grid_size
    # Start with a small radius to ensure no initial overlap.
    initial_radius = spacing / 2.1
    
    x0 = np.zeros(n * 3)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx < n:
                x0[idx*3]     = (i + 0.5) * spacing
                x0[idx*3 + 1] = (j + 0.5) * spacing
                x0[idx*3 + 2] = initial_radius
                idx += 1

    # Bounds for each variable: 0<=x,y<=1 and 0<=r<=0.5
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])

    # Define constraints for the optimizer.
    cons = [{'type': 'ineq', 'fun': constraints_func}]

    # Set a random seed for reproducibility. Although SLSQP is deterministic
    # from a given x0, this is good practice for complex numerical routines.
    np.random.seed(42)
    
    # Run the SLSQP optimizer.
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-7, 'disp': False}
    )

    if result.success:
        circles = result.x.reshape((n, 3))
        return circles
    else:
        # In case of optimization failure, return a zero-array as a fallback,
        # indicating no valid packing was found.
        return np.zeros((n, 3))

# EVOLVE-BLOCK-END
