# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint # Added Bounds, NonlinearConstraint for cleaner API
from numba import njit # Added Numba for JIT compilation of constraint calculations
# Removed pdist, squareform as Numba-compiled functions will replace their usage for constraints

def circle_packing32()->np.ndarray:
    """
    Finds an optimal arrangement of 32 non-overlapping circles in a unit square
    to maximize the sum of their radii using a constrained optimization approach.

    The problem is formulated for a Sequential Least Squares Programming (SLSQP)
    optimizer. It starts from a jittered grid configuration and iteratively adjusts
    the circles' positions (x, y) and radii (r) to maximize the sum of radii
    while satisfying boundary and non-overlap constraints.

    Returns:
        np.ndarray: An array of shape (32, 3) where each row represents a
                    circle as [x_center, y_center, radius].
    """
    n_circles = 32

    # Numba-compiled helper functions for performance (adapted from Inspiration Programs 1 & 3)
    @njit(fastmath=True)
    def _get_overlap_constraint_values(circles: np.ndarray) -> np.ndarray:
        """
        Calculates non-overlap constraint values for all unique pairs of circles.
        Constraint: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        Returns an array where each element is dist_sq - r_sum_sq.
        """
        n = circles.shape[0]
        num_pairs = n * (n - 1) // 2
        violations = np.empty(num_pairs, dtype=circles.dtype)
        k = 0
        for i in range(n):
            xi, yi, ri = circles[i, 0], circles[i, 1], circles[i, 2]
            for j in range(i + 1, n):
                xj, yj, rj = circles[j, 0], circles[j, 1], circles[j, 2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                r_sum_sq = (ri + rj)**2
                violations[k] = dist_sq - r_sum_sq
                k += 1
        return violations

    @njit(fastmath=True)
    def _get_boundary_constraint_values(circles: np.ndarray) -> np.ndarray:
        """
        Calculates boundary constraint values for all circles.
        Constraints: xi - ri >= 0, 1 - xi - ri >= 0, yi - ri >= 0, 1 - yi - ri >= 0
        Returns an array of 4*n constraint values.
        """
        n = circles.shape[0]
        violations = np.empty(4 * n, dtype=circles.dtype)
        for i in range(n):
            xi, yi, ri = circles[i, 0], circles[i, 1], circles[i, 2]
            violations[4*i + 0] = xi - ri       # x_i - r_i >= 0
            violations[4*i + 1] = 1 - xi - ri   # 1 - x_i - r_i >= 0
            violations[4*i + 2] = yi - ri       # y_i - r_i >= 0
            violations[4*i + 3] = 1 - yi - ri   # 1 - y_i - r_i >= 0
        return violations

    def _generate_initial_guess(n_circles: int, seed: int) -> np.ndarray:
        """
        Generates an initial configuration for circle packing.
        Uses a 6x6 grid, samples 32 points, jitters them, and assigns a
        bimodal distribution of initial radii to encourage size diversity.
        (Adapted from Inspiration Programs 1 & 3 for proven performance.)
        """
        rng = np.random.default_rng(seed)

        # Use a more square-like 6x6 grid and sample 32 points to break symmetry.
        grid_dim = 6
        n_grid_points = grid_dim * grid_dim
        
        # Calculate a base radius for the grid.
        base_radius = 1 / (2 * grid_dim) # Approx 1/12 = 0.083
        
        # Generate grid coordinates
        x_coords = np.linspace(base_radius, 1 - base_radius, grid_dim)
        y_coords = np.linspace(base_radius, 1 - base_radius, grid_dim)
        xx, yy = np.meshgrid(x_coords, y_coords)
        all_centers = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Randomly select 32 out of 36 grid points
        indices = rng.choice(n_grid_points, n_circles, replace=False)
        initial_centers = all_centers[indices]
        
        # Add jitter to further break symmetry, scaled by the base radius
        jitter_magnitude = base_radius * 0.25
        # Increased jitter magnitude to 0.35 * base_radius for more initial diversity
        jitter_magnitude = base_radius * 0.35
        initial_centers += rng.uniform(-jitter_magnitude, jitter_magnitude, initial_centers.shape)
        
        # Ensure centers stay within bounds after jittering
        initial_centers = np.clip(initial_centers, 0.0, 1.0)
        
        # Introduce a bimodal distribution for initial radii to encourage size diversity.
        initial_radii = np.zeros(n_circles)
        num_large_circles = n_circles // 4  # Designate ~1/4 of circles to be potentially larger
        large_indices = rng.choice(n_circles, num_large_circles, replace=False)
        small_indices = np.setdiff1d(np.arange(n_circles), large_indices)

        # Larger circles can start up to 1.8x base_radius
        initial_radii[large_indices] = rng.uniform(base_radius * 0.8, base_radius * 1.8, num_large_circles)
        # Smaller circles fill the gaps
        initial_radii[small_indices] = rng.uniform(base_radius * 0.4, base_radius * 1.0, n_circles - num_large_circles)
        
        # Flatten parameters into a 1D array for the optimizer: [x0,y0,r0, x1,y1,r1, ...].
        x0 = np.hstack([initial_centers, initial_radii.reshape(-1, 1)]).ravel()
        return x0

    # Multi-start optimization setup
    # Increased restarts to 50, leveraging Numba speedup for broader search
    num_restarts = 50 
    initial_seed = 42 # Base seed for deterministic sequence of restarts

    best_circles = None
    best_sum_radii = -np.inf # Objective minimizes negative sum_radii, so we maximize this
    
    # Store an initial guess for fallback if all restarts fail
    x0_fallback = _generate_initial_guess(n_circles, initial_seed)

    # 2. Objective Function (to be minimized).
    def objective(params: np.ndarray) -> float:
        """The objective is to maximize the sum of radii, so we minimize its negative."""
        radii = params[2::3]
        return -np.sum(radii)

    # 3. Constraint Function (all returned values must be >= 0 for feasibility).
    def constraints(params: np.ndarray) -> np.ndarray:
        """
        Returns a single array of all constraint values, using Numba-compiled helpers.
        - Boundary constraints: 4 * n_circles
        - Non-overlap constraints: n_circles * (n_circles - 1) / 2
        """
        circles = params.reshape((n_circles, 3))

        boundary_vals = _get_boundary_constraint_values(circles)
        overlap_vals = _get_overlap_constraint_values(circles)
        
        return np.concatenate([boundary_vals, overlap_vals])

    # Define bounds once. A small positive lower bound on radius prevents degenerate cases.
    min_radius = 1e-6 # Consistent with Inspiration 1 for numerical stability, slightly smaller than 0.001
    bounds = Bounds(
        np.tile([0.0, 0.0, min_radius], n_circles),
        np.tile([1.0, 1.0, 0.5], n_circles)
    )
    # Define the nonlinear constraint object once, for efficiency with multiple restarts (adapted from Inspiration Programs 1 & 3)
    nonlinear_constraint = NonlinearConstraint(constraints, 0, np.inf)

    # 5. Run the optimizer multiple times with different initial guesses.
    for i in range(num_restarts):
        current_seed = initial_seed + i # Generate unique seed for each restart
        x0 = _generate_initial_guess(n_circles, current_seed)

        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[nonlinear_constraint], # Pass the NonlinearConstraint object (adapted from Inspiration Programs 1 & 3)
            # Increased maxiter to 5000 for more thorough convergence
            options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-9} 
        )

        if res.success:
            current_sum_radii = -res.fun # objective returns negative sum_radii
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = res.x.reshape((n_circles, 3))
                # Optional: print progress for debugging
                # print(f"Restart {i+1}: New best sum_radii = {best_sum_radii:.4f}")
        # else:
            # Optional: print warnings for failed restarts
            # print(f"Restart {i+1}: Optimization failed: {res.message}")
    
    # 6. Process and return the best result found across all restarts.
    if best_circles is not None:
        circles = best_circles
        # A final check for constraint satisfaction (minor violations can occur due to ftol)
        final_params = circles.ravel()
        if np.any(constraints(final_params) < -1e-7): # Use a slightly stricter tolerance for final check
            print("Warning: The best found solution shows minor constraint violations. This might indicate issues with ftol or maxiter.")
    else:
        # Fallback to the initial guess from the base seed if all optimizations fail.
        print(f"Warning: All {num_restarts} optimizations failed. Returning the initial guess from seed {initial_seed}.")
        circles = x0_fallback.reshape((n_circles, 3))

    return circles

# EVOLVE-BLOCK-END
