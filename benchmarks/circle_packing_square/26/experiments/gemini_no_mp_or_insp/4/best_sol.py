# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import pdist

def circle_packing26()->np.ndarray:
    """
    Finds an optimal arrangement of 26 non-overlapping circles in a unit square
    to maximize the sum of their radii using a hybrid global-local optimization strategy.
    It combines `scipy.optimize.basinhopping` for global search with `scipy.optimize.minimize`
    (SLSQP method) for local refinement.

    This approach defines the problem with 78 variables (x, y, r for each of the 26 circles)
    and a set of non-linear constraints for containment and non-overlapping.
    `basinhopping` performs multiple local optimizations from perturbed starting points
    to explore a wider range of the solution space and improve the chances of finding
    a better local optimum, potentially closer to the global optimum.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the
                 (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    # Use a fixed seed for reproducibility. basinhopping also takes a seed argument.
    rng_seed = 42

    def get_initial_guess(n_circles, grid_rows, grid_cols, seed):
        """
        Generates a structured initial guess by selecting points from a perturbed grid.
        Uses a provided seed for reproducibility of the structured guess.
        """
        x_coords = np.linspace(0, 1, grid_cols + 2)[1:-1]
        y_coords = np.linspace(0, 1, grid_rows + 2)[1:-1]
        
        grid_points = np.array([(x, y) for y in y_coords for x in x_coords])
        
        # Randomly select n_circles points from the grid to break symmetry.
        local_rng = np.random.default_rng(seed) 
        selected_indices = local_rng.choice(len(grid_points), n_circles, replace=False)
        selected_points = grid_points[selected_indices]
        
        # Start with a safe, small radius, slightly adjusted.
        r_init = min(1 / (2 * grid_cols), 1 / (2 * grid_rows)) * 0.9
        
        x0 = np.zeros(n_circles * 3)
        for i, (x, y) in enumerate(selected_points):
            idx = i * 3
            # Add small random perturbation to positions
            x0[idx] = x + local_rng.uniform(-0.01, 0.01)
            x0[idx+1] = y + local_rng.uniform(-0.01, 0.01)
            x0[idx+2] = r_init
        return x0

    # The objective is to maximize sum of radii, which is equivalent to
    # minimizing the negative sum of radii.
    def objective(vars_flat):
        return -np.sum(vars_flat[2::3])

    # Pre-compute indices for efficient pairwise calculations in the constraint function.
    # These correspond to the upper triangle of the pairwise matrix.
    rows, cols = np.triu_indices(n, k=1)

    def constraints_all(vars_flat):
        """
        Vectorized constraint function. Returns a single array where each element
        must be non-negative for the solution to be feasible.
        """
        # Ensure radii are positive for constraint calculations, even if bounds theoretically prevent negative.
        # This helps numerical stability if optimization tries to push radii to exactly zero or slightly negative.
        vars_reshaped = vars_flat.reshape((n, 3))
        x, y = vars_reshaped[:, 0], vars_reshaped[:, 1]
        r = np.maximum(vars_reshaped[:, 2], 1e-8) # A very small positive floor for radius

        # 1. Boundary constraints (4 * n): ri <= xi, xi <= 1-ri, etc.
        boundary_c = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

        # 2. Non-overlap constraints (n * (n-1) / 2):
        # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        # To avoid sqrt, we use: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        coords = vars_reshaped[:, :2] # Use original coords, r is already adjusted
        dist_sq = pdist(coords)**2
        
        r_i = r[rows]
        r_j = r[cols]
        radii_sum_sq = (r_i + r_j)**2
        
        overlap_c = dist_sq - radii_sum_sq
        
        return np.concatenate([boundary_c, overlap_c])

    # --- Optimizer Setup ---
    # Common bounds for all variables (x, y in [0,1], r in [1e-6, 0.5])
    # Radii must be positive, so a small lower bound like 1e-6 is preferred.
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]) 

    # Arguments for the local minimizer (SLSQP) within basinhopping
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": {'type': 'ineq', 'fun': constraints_all},
        "options": {'maxiter': 2000, 'ftol': 1e-10, 'disp': False} # Thorough local search
    }

    num_bh_runs = 3 # Number of independent basinhopping runs with diverse initial guesses
    best_sum_radii = -np.inf
    best_circles = None

    for run_idx in range(num_bh_runs):
        # Generate diverse initial guess for each basinhopping run
        current_run_seed = rng_seed + run_idx
        local_rng = np.random.default_rng(current_run_seed)

        if run_idx == 0:
            # First run uses the structured grid-based guess
            x0 = get_initial_guess(n_circles=n, grid_rows=5, grid_cols=6, seed=current_run_seed)
        else:
            # Subsequent runs use random initial guesses
            x_init = local_rng.uniform(0.05, 0.95, size=n) # x-coords
            y_init = local_rng.uniform(0.05, 0.95, size=n) # y-coords
            r_init = local_rng.uniform(0.01, 0.05, size=n) # radii, small to avoid immediate severe overlaps

            x0 = np.zeros(n * 3)
            x0[0::3] = x_init
            x0[1::3] = y_init
            x0[2::3] = r_init

        # Run basinhopping for global exploration
        res = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,      # Number of hopping iterations per run
            T=1.0,          # Temperature parameter
            stepsize=0.05,  # Step size for perturbations
            seed=current_run_seed, # Use a different seed for each BH run for diverse internal randomness
            disp=False      # Set to True for verbose output
        )

        current_sum_radii = -res.fun
        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = res.x.reshape((n, 3))
            # print(f"Run {run_idx+1}: New best sum_radii = {best_sum_radii:.6f}")

    if best_circles is None:
        # Fallback if no viable solution found after multiple runs.
        print("Warning: No viable solution found after multiple runs. Returning default zeros.")
        return np.zeros((n,3))
    
    # Final cleanup: Ensure radii are positive and positions are strictly within bounds.
    # This acts as a safeguard against minor numerical deviations from the optimizer.
    final_circles = best_circles
    final_circles[:, 2] = np.maximum(final_circles[:, 2], 1e-6) # Ensure radii are truly positive
    final_circles[:, 0] = np.clip(final_circles[:, 0], final_circles[:, 2], 1 - final_circles[:, 2])
    final_circles[:, 1] = np.clip(final_circles[:, 1], final_circles[:, 2], 1 - final_circles[:, 2])

    return final_circles

# EVOLVE-BLOCK-END