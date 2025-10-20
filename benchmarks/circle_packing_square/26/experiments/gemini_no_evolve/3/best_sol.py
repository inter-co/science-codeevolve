# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by formulating the problem as a constrained optimization and solving it with SLSQP.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    
    # Set a seed for reproducibility of the initial guess
    np.random.seed(42)

    # Objective function: maximize sum of radii -> minimize -sum(radii)
    def objective(params):
        # Radii are the last N_CIRCLES elements of the parameter vector
        radii = params[2 * N_CIRCLES:]
        return -np.sum(radii)

    # Constraints function (all constraints must be >= 0)
    # This is vectorized for performance.
    def constraints_all(params):
        xs = params[:N_CIRCLES]
        ys = params[N_CIRCLES:2 * N_CIRCLES]
        rs = params[2 * N_CIRCLES:]

        # 1. Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
        # Re-written as: (x-r >= 0), (1-x-r >= 0), (y-r >= 0), (1-y-r >= 0)
        c_boundary = np.concatenate([
            xs - rs,
            1 - xs - rs,
            ys - rs,
            1 - ys - rs
        ])

        # 2. Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        # Re-written for efficiency as: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        x_col = xs[:, np.newaxis]
        y_col = ys[:, np.newaxis]
        r_col = rs[:, np.newaxis]

        dist_sq = (x_col - x_col.T)**2 + (y_col - y_col.T)**2
        r_sum_sq = (r_col + r_col.T)**2

        # Get upper triangle indices to avoid duplicate constraints (i,j) vs (j,i)
        # and self-constraints (i,i)
        iu = np.triu_indices(N_CIRCLES, k=1)
        c_overlap = dist_sq[iu] - r_sum_sq[iu]

        return np.concatenate([c_boundary, c_overlap])

    # Initial guess: A slightly perturbed grid with non-uniform initial radii.
    # This provides a reasonable starting distribution for positions.
    rows, cols = 6, 5
    x_points = np.linspace(1 / (2 * cols), 1 - 1 / (2 * cols), cols)
    y_points = np.linspace(1 / (2 * rows), 1 - 1 / (2 * rows), rows)
    xx, yy = np.meshgrid(x_points, y_points)
    
    # Add a small random jitter to break symmetry and explore better configurations
    jitter_x = np.random.uniform(-0.01, 0.01, xx.shape)
    jitter_y = np.random.uniform(-0.01, 0.01, yy.shape)

    initial_x = np.clip((xx + jitter_x).ravel()[:N_CIRCLES], 0.01, 0.99)
    initial_y = np.clip((yy + jitter_y).ravel()[:N_CIRCLES], 0.01, 0.99)

    # Create a non-uniform initial radius distribution.
    # Circles further from the center (0.5, 0.5) get a larger initial radius.
    # This encourages the optimizer to explore solutions with larger boundary circles,
    # a common feature in optimal finite circle packing solutions.
    dist_from_center = np.sqrt((initial_x - 0.5)**2 + (initial_y - 0.5)**2)
    max_dist = np.sqrt(0.5) # Max possible distance from center in a unit square
    normalized_dist = dist_from_center / max_dist

    # Assign radii based on this normalized distance, e.g., from 0.02 to 0.05
    min_r, max_r = 0.02, 0.05
    initial_r = min_r + (max_r - min_r) * normalized_dist

    x0 = np.concatenate([initial_x, initial_y, initial_r])

    # Bounds for each variable (x, y, r)
    bounds_x = [(0.0, 1.0)] * N_CIRCLES
    bounds_y = [(0.0, 1.0)] * N_CIRCLES
    bounds_r = [(0.0, 0.5)] * N_CIRCLES # A single circle can have at most r=0.5
    bounds = bounds_x + bounds_y + bounds_r

    # Constraint dictionary for SLSQP
    cons = ({'type': 'ineq', 'fun': constraints_all})

    # Run the optimization.
    # Increased maxiter for better convergence from the more promising initial guess.
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False})

    # Reshape the result into the (N, 3) circle format
    solution = result.x
    
    final_xs = solution[:N_CIRCLES]
    final_ys = solution[N_CIRCLES:2 * N_CIRCLES]
    final_rs = solution[2 * N_CIRCLES:]

    circles = np.stack([final_xs, final_ys, final_rs], axis=1)

    return circles

# EVOLVE-BLOCK-END