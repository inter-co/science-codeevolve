# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import numba

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is a complex non-convex optimization problem. We use a Sequential Least Squares
    Programming (SLSQP) algorithm to find a solution.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y)
                 coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # The objective is to maximize the sum of radii, which is equivalent to
    # minimizing the negative sum of radii.
    def objective(params: np.ndarray) -> float:
        """The objective function to minimize (negative sum of radii)."""
        # params is a flat array: [x0, y0, r0, x1, y1, r1, ...]
        radii = params[2::3]
        return -np.sum(radii)

    # Use Numba to JIT-compile the constraint evaluation for a significant speedup.
    # This function is the performance bottleneck.
    @numba.jit(nopython=True, fastmath=True)
    def fast_constraints(params: np.ndarray, n: int):
        """
        Calculates all constraint values. For SLSQP, constraints are satisfied
        if their value is non-negative (>= 0).
        """
        positions = params.reshape((n, 3))[:, :2]
        radii = params.reshape((n, 3))[:, 2]

        # 1. Boundary constraints: 4*n constraints
        # c_i >= 0 for all i
        boundary_c = np.empty(4 * n)
        for i in range(n):
            boundary_c[i]       = positions[i, 0] - radii[i]          # x_i - r_i >= 0
            boundary_c[i + n]   = 1.0 - positions[i, 0] - radii[i]    # 1 - x_i - r_i >= 0
            boundary_c[i + 2*n] = positions[i, 1] - radii[i]          # y_i - r_i >= 0
            boundary_c[i + 3*n] = 1.0 - positions[i, 1] - radii[i]    # 1 - y_i - r_i >= 0

        # 2. Non-overlap constraints: n*(n-1)/2 constraints
        # dist(ci, cj) >= ri + rj  <=>  dist^2 >= (ri+rj)^2
        num_pairs = n * (n - 1) // 2
        overlap_c = np.empty(num_pairs)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2
                radii_sum = radii[i] + radii[j]
                overlap_c[k] = dist_sq - radii_sum**2
                k += 1
        
        return np.concatenate((boundary_c, overlap_c))

    # Scipy's minimize function requires a wrapper that matches its signature.
    def constraints_wrapper(params: np.ndarray) -> np.ndarray:
        return fast_constraints(params, n)

    # A good initial guess is crucial for the optimizer to find a high-quality solution.
    # We start with a grid, but introduce non-uniform radii to break symmetry
    # and guide the optimizer towards a more optimal, non-uniform solution.
    rng = np.random.default_rng(42) # Use a fixed seed for reproducibility.
    
    # Arrange 26 circles in a 6x5 grid pattern, which leaves 4 empty spots.
    rows, cols = 6, 5
    x_centers = np.linspace(1/(2*cols), 1 - 1/(2*cols), cols)
    y_centers = np.linspace(1/(2*rows), 1 - 1/(2*rows), rows)
    xx, yy = np.meshgrid(x_centers, y_centers)
    
    initial_pos = np.vstack([xx.ravel(), yy.ravel()]).T
    initial_pos = initial_pos[:n]
    # Add a slightly larger jitter to further break the grid symmetry.
    initial_pos += rng.normal(scale=0.01, size=initial_pos.shape)

    # Create a non-uniform initial radius distribution. Circles closer to the
    # center of the square [0.5, 0.5] are given a larger initial radius.
    center = np.array([0.5, 0.5])
    distances_from_center = np.linalg.norm(initial_pos - center, axis=1)
    
    # Base radius from the grid spacing, slightly smaller to give optimizer room.
    base_radius = 0.5 * min(1/cols, 1/rows) * 0.9
    
    # Radii are modulated by distance from center (closer = larger).
    # This creates a spread of radii, encouraging a more varied final packing.
    max_dist = np.max(distances_from_center)
    if max_dist > 0:
        norm_dist = distances_from_center / max_dist
        # Radii range from 1.15*base (center) to 0.85*base (edge)
        modulation = 1.15 - 0.30 * norm_dist
    else: # Handle case where all points are at the same distance
        modulation = 1.0
    initial_radii = base_radius * modulation
    
    # Clip initial positions to be within bounds given the new radii, as jitter
    # might cause slight violations. This ensures a feasible start.
    initial_pos[:, 0] = np.clip(initial_pos[:, 0], initial_radii, 1 - initial_radii)
    initial_pos[:, 1] = np.clip(initial_pos[:, 1], initial_radii, 1 - initial_radii)

    x0 = np.hstack([initial_pos, initial_radii.reshape(-1, 1)]).flatten()

    # Define bounds for each variable: 0 <= x,y <= 1 and 0 <= r <= 0.5
    bounds_list = []
    for i in range(n):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])

    # Define the list of constraints for the optimizer.
    cons = [{'type': 'ineq', 'fun': constraints_wrapper}]

    # Run the SLSQP optimizer.
    # This is a computationally intensive step.
    # Run the SLSQP optimizer with more iterations and tighter tolerance for a higher quality solution.
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 2500, 'disp': False, 'ftol': 1e-10}
    )

    if result.success:
        circles = result.x.reshape((n, 3))
    else:
        # If optimization fails, return the last state, which may be partially optimized.
        # A warning is useful for debugging in a real-world scenario.
        # print(f"Warning: Optimization failed to converge. Reason: {result.message}")
        circles = result.x.reshape((n, 3))
    
    return circles

# EVOLVE-BLOCK-END