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

    # Multi-start optimization setup
    num_restarts = 20 # Increased restarts for broader search of the solution space
    initial_seed = 42 # Base seed for deterministic sequence of restarts

    best_circles = None
    best_sum_radii = -np.inf # Objective minimizes negative sum_radii, so we maximize this
    
    # Store an initial guess for fallback if all restarts fail
    # x0_fallback will be initialized after the final _generate_initial_guess definition

    # Numba-compiled helper functions for performance (adapted from Inspiration 3)
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

    # Define a consistent minimum radius for all initial guess generators and bounds.
    min_radius = 1e-6 # Consistent with Inspiration 1 for numerical stability

    # --- Initial Population Generation Helper Functions (adapted from Inspiration 2 and target) ---

    def _generate_jittered_grid_bimodal_radii(n_circles: int, seed: int) -> np.ndarray:
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

    def _generate_hex_grid_pattern(n_circles: int, seed: int, square_size: float = 1.0, perturbation_scale: float = 0.005) -> np.ndarray:
        """
        Generates an initial configuration of circles in a hexagonal grid pattern,
        centered within the unit square.
        """
        rng = np.random.default_rng(seed)
        circles_temp = []
        
        # Estimate radius based on square area and target number of circles, adjusted for hex packing
        # A common heuristic: r = 0.5 / (sqrt(n_circles/0.866) + 1)
        r_estimate = 0.5 / (np.sqrt(n_circles / (np.sqrt(3)/2)) + 1) # Adjusted for hex density
        r_estimate = np.clip(r_estimate, min_radius, 0.15)

        dx = 2 * r_estimate
        dy = np.sqrt(3) * r_estimate
        row_idx = 0
        
        while True:
            y_center_uncentered = r_estimate + row_idx * dy
            if y_center_uncentered + r_estimate > square_size + 1e-9:
                break

            col_idx = 0
            while True:
                x_offset = 0.0 if row_idx % 2 == 0 else r_estimate
                x_center_uncentered = r_estimate + col_idx * dx + x_offset
                
                if x_center_uncentered + r_estimate > square_size + 1e-9:
                    break
                
                circles_temp.append([x_center_uncentered, y_center_uncentered, r_estimate])
                
                if len(circles_temp) >= n_circles:
                    break
                col_idx += 1
            
            if len(circles_temp) >= n_circles:
                break
            row_idx += 1

        circles_temp_np = np.array(circles_temp)
        
        if len(circles_temp_np) == 0:
            return np.array([[0.5, 0.5, min_radius]] * n_circles).ravel()

        if len(circles_temp_np) > n_circles:
            # Randomly select N circles from the generated grid to break some symmetry.
            indices = rng.choice(len(circles_temp_np), n_circles, replace=False)
            circles_data = circles_temp_np[indices]
        elif len(circles_temp_np) < n_circles:
            circles_data = circles_temp_np
            # Fill remaining with random small circles
            while len(circles_data) < n_circles:
                rand_r = rng.uniform(min_radius, r_estimate * 0.5)
                x_rand = rng.uniform(rand_r, square_size - rand_r)
                y_rand = rng.uniform(rand_r, square_size - rand_r)
                circles_data = np.vstack([circles_data, [x_rand, y_rand, rand_r]])
        else:
            circles_data = circles_temp_np

        # Center the grid
        min_x_centers = np.min(circles_data[:, 0])
        max_x_centers = np.max(circles_data[:, 0])
        min_y_centers = np.min(circles_data[:, 1])
        max_y_centers = np.max(circles_data[:, 1])

        # Adjust for the actual radii used in the grid to calculate true width/height
        current_grid_width = (max_x_centers - min_x_centers) + 2 * r_estimate
        current_grid_height = (max_y_centers - min_y_centers) + 2 * r_estimate

        shift_x = (square_size - current_grid_width) / 2.0 - (min_x_centers - r_estimate)
        shift_y = (square_size - current_grid_height) / 2.0 - (min_y_centers - r_estimate)
        
        circles_data[:, 0] += shift_x
        circles_data[:, 1] += shift_y

        # Add perturbation
        if perturbation_scale > 0:
            circles_data[:, :2] += rng.uniform(-perturbation_scale, perturbation_scale, (n_circles, 2))
            circles_data[:, 2] += rng.uniform(-perturbation_scale * 0.5, perturbation_scale * 0.5, n_circles)

        circles_data[:, 0] = np.clip(circles_data[:, 0], circles_data[:, 2], square_size - circles_data[:, 2])
        circles_data[:, 1] = np.clip(circles_data[:, 1], circles_data[:, 2], square_size - circles_data[:, 2])
        circles_data[:, 2] = np.clip(circles_data[:, 2], min_radius, 0.5)

        return circles_data.ravel()

    def _generate_square_grid_pattern(n_circles: int, seed: int, square_size: float = 1.0, perturbation_scale: float = 0.005) -> np.ndarray:
        """
        Generates an initial configuration of circles in a square grid pattern,
        centered within the unit square.
        """
        rng = np.random.default_rng(seed)
        circles_temp = []
        
        n_rows_cols = int(np.ceil(np.sqrt(n_circles)))
        r_estimate = square_size / (2 * n_rows_cols)
        r_estimate = np.clip(r_estimate, min_radius, 0.15)
        
        current_r = r_estimate
        step = 2 * current_r
        
        for row_idx in range(n_rows_cols):
            for col_idx in range(n_rows_cols):
                x_center = current_r + col_idx * step
                y_center = current_r + row_idx * step
                
                if x_center + current_r <= square_size + 1e-9 and \
                   y_center + current_r <= square_size + 1e-9 and \
                   len(circles_temp) < n_circles:
                    circles_temp.append([x_center, y_center, current_r])
                
                if len(circles_temp) >= n_circles:
                    break
            if len(circles_temp) >= n_circles:
                break

        circles_temp_np = np.array(circles_temp)

        if len(circles_temp_np) == 0:
            return np.array([[0.5, 0.5, min_radius]] * n_circles).ravel()

        if len(circles_temp_np) > n_circles:
            indices = rng.choice(len(circles_temp_np), n_circles, replace=False)
            circles_data = circles_temp_np[indices]
        elif len(circles_temp_np) < n_circles:
            circles_data = circles_temp_np
            while len(circles_data) < n_circles:
                rand_r = rng.uniform(min_radius, r_estimate * 0.5)
                x_rand = rng.uniform(rand_r, square_size - rand_r)
                y_rand = rng.uniform(rand_r, square_size - rand_r)
                circles_data = np.vstack([circles_data, [x_rand, y_rand, rand_r]])
        else:
            circles_data = circles_temp_np
        
        # Center the grid
        min_x_centers = np.min(circles_data[:, 0])
        max_x_centers = np.max(circles_data[:, 0])
        min_y_centers = np.min(circles_data[:, 1])
        max_y_centers = np.max(circles_data[:, 1])

        current_grid_width = (max_x_centers - min_x_centers) + 2 * current_r
        current_grid_height = (max_y_centers - min_y_centers) + 2 * current_r

        shift_x = (square_size - current_grid_width) / 2.0 - (min_x_centers - current_r)
        shift_y = (square_size - current_grid_height) / 2.0 - (min_y_centers - current_r)
        
        circles_data[:, 0] += shift_x
        circles_data[:, 1] += shift_y

        # Add perturbation
        if perturbation_scale > 0:
            circles_data[:, :2] += rng.uniform(-perturbation_scale, perturbation_scale, (n_circles, 2))
            circles_data[:, 2] += rng.uniform(-perturbation_scale * 0.5, perturbation_scale * 0.5, n_circles)

        circles_data[:, 0] = np.clip(circles_data[:, 0], circles_data[:, 2], square_size - circles_data[:, 2])
        circles_data[:, 1] = np.clip(circles_data[:, 1], circles_data[:, 2], square_size - circles_data[:, 2])
        circles_data[:, 2] = np.clip(circles_data[:, 2], min_radius, 0.5)

        return circles_data.ravel()

    def _generate_anchored_random_pattern(n_circles: int, seed: int, square_size: float = 1.0, perturbation_scale: float = 0.005) -> np.ndarray:
        """
        Generates an initial configuration with a few large anchor circles
        (e.g., in corners and center) and the rest as small random fillers.
        This introduces a multi-scale packing characteristic and exploits boundary advantages.
        """
        rng = np.random.default_rng(seed)
        circles_data = np.zeros((n_circles, 3))
        
        num_anchors = 0
        
        r_large_corner_candidate = 0.12 
        if n_circles >= 4:
            r_corner = np.clip(r_large_corner_candidate, min_radius, 0.5)
            circles_data[0] = [r_corner, r_corner, r_corner]
            circles_data[1] = [square_size - r_corner, r_corner, r_corner]
            circles_data[2] = [r_corner, square_size - r_corner, r_corner]
            circles_data[3] = [square_size - r_corner, square_size - r_corner, r_corner]
            num_anchors += 4

        r_center_candidate = 0.15 
        if n_circles >= 5:
            r_center = np.clip(r_center_candidate, min_radius, 0.5)
            circles_data[num_anchors] = [square_size / 2.0, square_size / 2.0, r_center]
            num_anchors += 1
        
        # Fill the rest with small random circles
        r_filler_max = r_large_corner_candidate * 0.3 # Max filler radius based on anchor size
        for i in range(num_anchors, n_circles):
            rand_r = rng.uniform(min_radius, r_filler_max)
            circles_data[i, 0] = rng.uniform(rand_r, square_size - rand_r)
            circles_data[i, 1] = rng.uniform(rand_r, square_size - rand_r)
            circles_data[i, 2] = rand_r
        
        # Add perturbation to all circles, including anchors
        if perturbation_scale > 0:
            circles_data[:, :2] += rng.uniform(-perturbation_scale, perturbation_scale, (n_circles, 2))
            circles_data[:, 2] += rng.uniform(-perturbation_scale * 0.5, perturbation_scale * 0.5, n_circles)

        # Re-clip after perturbation
        for i in range(n_circles):
            r_val = circles_data[i, 2]
            circles_data[i, 0] = np.clip(circles_data[i, 0], r_val, square_size - r_val)
            circles_data[i, 1] = np.clip(circles_data[i, 1], r_val, square_size - r_val)
            circles_data[i, 2] = np.clip(r_val, min_radius, 0.5)
            
        return circles_data.ravel()

    def _generate_random_dense_pattern(n_circles: int, seed: int, square_size: float = 1.0, perturbation_scale: float = 0.005) -> np.ndarray:
        """
        Generates an initial configuration with randomly placed circles,
        with radii in a relatively narrow, small range to encourage dense packing.
        """
        rng = np.random.default_rng(seed)
        circles_data = np.zeros((n_circles, 3))
        
        # Vary max radius for diversity among random patterns
        rand_r_max_base = 0.06
        rand_r_max = rng.uniform(rand_r_max_base * 0.8, rand_r_max_base * 1.2)
        
        for j in range(n_circles):
            rand_r = rng.uniform(min_radius, rand_r_max)
            circles_data[j, 0] = rng.uniform(rand_r, square_size - rand_r)
            circles_data[j, 1] = rng.uniform(rand_r, square_size - rand_r)
            circles_data[j, 2] = rand_r
        
        # Add perturbation
        if perturbation_scale > 0:
            circles_data[:, :2] += rng.uniform(-perturbation_scale, perturbation_scale, (n_circles, 2))
            circles_data[:, 2] += rng.uniform(-perturbation_scale * 0.5, perturbation_scale * 0.5, n_circles)
        
        # Re-clip after perturbation
        for i in range(n_circles):
            r_val = circles_data[i, 2]
            circles_data[i, 0] = np.clip(circles_data[i, 0], r_val, square_size - r_val)
            circles_data[i, 1] = np.clip(circles_data[i, 1], r_val, square_size - r_val)
            circles_data[i, 2] = np.clip(r_val, min_radius, 0.5)

        return circles_data.ravel()

    def _generate_initial_guess_dispatcher(n_circles: int, restart_idx: int, initial_seed: int) -> np.ndarray:
        """
        Dispatches to different initial guess generation strategies based on the restart index.
        This provides a diverse set of starting points for the multi-start optimization.
        """
        # Use a combination of initial_seed and restart_idx for unique seeds
        current_seed = initial_seed + restart_idx
        
        # Define a list of initial guess strategies
        strategies = [
            _generate_jittered_grid_bimodal_radii,
            _generate_hex_grid_pattern,
            _generate_square_grid_pattern,
            _generate_anchored_random_pattern,
            _generate_random_dense_pattern
        ]
        
        # Cycle through strategies
        strategy_func = strategies[restart_idx % len(strategies)]
        
        return strategy_func(n_circles, current_seed)

    # Store an initial guess for fallback if all restarts fail
    x0_fallback = _generate_initial_guess_dispatcher(n_circles, 0, initial_seed) # Use the first strategy for fallback

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
    min_radius = 1e-6 # Consistent with Inspiration 1 for numerical stability
    bounds = Bounds(
        np.tile([0.0, 0.0, min_radius], n_circles),
        np.tile([1.0, 1.0, 0.5], n_circles)
    )
    # Define the nonlinear constraint object once, for efficiency with multiple restarts
    nonlinear_constraint = NonlinearConstraint(constraints, 0, np.inf)

    # Increase num_restarts to fully leverage the diverse initializations.
    num_restarts = 40 # Original 20, increased to 40 for more exploration with 5 strategies.
    
    # Run the optimizer multiple times with different initial guesses
    for i in range(num_restarts):
        # Generate initial guess using the dispatcher, cycling through strategies
        x0 = _generate_initial_guess_dispatcher(n_circles, i, initial_seed)

        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[nonlinear_constraint], # Pass the NonlinearConstraint object
            # Use tight tolerance, high maxiter, and add 'eps' from Inspiration 1 for high-quality solutions
            options={'maxiter': 4000, 'ftol': 1e-11, 'disp': False, 'eps': 1e-9} # Increased maxiter, tightened ftol
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
