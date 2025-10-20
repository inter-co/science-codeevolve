# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit # Import numba for JIT compilation

def circle_packing26() -> np.ndarray:
    """
    Finds an optimal arrangement of 26 non-overlapping circles within a unit square
    to maximize the sum of their radii, using a hybrid optimization approach:
    multiple SLSQP runs with varied initial guesses, with Numba for performance.

    This function formulates the problem as a non-linear constrained optimization
    and uses a gradient-based solver to find a high-quality local optimum.

    Returns:
        np.ndarray: Array of shape (26, 3) with [x, y, r] for each circle.
    """
    n_circles = 26
    seed = 42 # Keep seed for reproducibility of random elements
    np.random.seed(seed)

    # The objective is to maximize the sum of radii, which is equivalent to
    # minimizing the negative sum of radii.
    def objective(params: np.ndarray) -> float:
        """Computes the negative sum of radii."""
        # params is a flat array: [x0, y0, r0, x1, y1, r1, ...]
        radii = params[2::3]
        return -np.sum(radii)

    # All constraints are formulated as g(x) >= 0.
    # Use numba for JIT compilation to speed up constraint evaluation.
    @njit
    def _constraints_numba(params: np.ndarray, n_circles: int) -> np.ndarray:
        """
        Numba-compiled helper for constraint calculations.
        This function computes containment and non-overlap constraints.
        """
        circles = params.reshape((n_circles, 3))
        positions = circles[:, :2]
        radii = circles[:, 2]

        # 1. Containment constraints:
        #    ri <= xi <= 1-ri  =>  xi - ri >= 0  AND  1 - xi - ri >= 0
        #    ri <= yi <= 1-ri  =>  yi - ri >= 0  AND  1 - yi - ri >= 0
        containment_constraints = np.empty(n_circles * 4, dtype=params.dtype)
        for i in range(n_circles):
            containment_constraints[i] = positions[i, 0] - radii[i]
            containment_constraints[n_circles + i] = 1.0 - positions[i, 0] - radii[i]
            containment_constraints[2 * n_circles + i] = positions[i, 1] - radii[i]
            containment_constraints[3 * n_circles + i] = 1.0 - positions[i, 1] - radii[i]

        # 2. Non-overlap constraints:
        #    (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        num_overlap_constraints = n_circles * (n_circles - 1) // 2
        overlap_constraints = np.empty(num_overlap_constraints, dtype=params.dtype)
        k = 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_sq = (positions[i, 0] - positions[j, 0])**2 + \
                          (positions[i, 1] - positions[j, 1])**2
                r_sum_sq = (radii[i] + radii[j])**2
                overlap_constraints[k] = dist_sq - r_sum_sq
                k += 1
        
        return np.concatenate((containment_constraints, overlap_constraints))

    # Wrapper function for scipy.optimize.minimize to call the numba-compiled function
    def constraints_func(params: np.ndarray) -> np.ndarray:
        return _constraints_numba(params, n_circles)

    # --- Optimization Setup ---
    # Bounds for each variable: 0 <= x, y <= 1 and 0.01 <= r <= 0.5
    bounds = []
    for _ in range(n_circles):
        bounds.extend([(0, 1), (0, 1), (0.01, 0.5)]) # Min radius to avoid collapse

    # Define constraints for the optimizer
    cons = {'type': 'ineq', 'fun': constraints_func}

    best_sum_radii = -np.inf
    best_final_circles = None

    # Number of initial guesses to try to explore more local optima
    num_initial_guesses = 5 

    for i in range(num_initial_guesses):
        # --- Initial Guess Generation ---
        # Vary initial guess generation strategy for different runs.
        # This helps SLSQP escape local minima by starting in different basins of attraction.
        
        if i == 0: # Strategy 1: Perturbed grid (similar to original, but larger perturbation)
            rows, cols = 5, 6
            x_coords = (np.arange(rows) + 0.5) / rows
            y_coords = (np.arange(cols) + 0.5) / cols
            xv, yv = np.meshgrid(x_coords, y_coords)
            initial_positions = np.vstack([xv.ravel(), yv.ravel()]).T[:n_circles]
            initial_positions += np.random.uniform(-0.02, 0.02, initial_positions.shape) # Larger perturbation
            initial_positions = np.clip(initial_positions, 0.01, 0.99)
            initial_radii = np.full(n_circles, 0.08)
        
        elif i == 1: # Strategy 2: Random initial placement with smaller radii
            initial_positions = np.random.uniform(0.1, 0.9, (n_circles, 2))
            initial_radii = np.random.uniform(0.03, 0.06, n_circles) # Smaller, random radii
        
        elif i == 2: # Strategy 3: Hexagonal-like pattern (rough approximation)
            # This attempts to create a denser initial arrangement.
            num_rows_approx = int(np.ceil(np.sqrt(n_circles / 0.8))) # Heuristic for number of rows
            avg_radius_estimate = 0.5 / num_rows_approx
            
            initial_positions_list = []
            for row in range(num_rows_approx):
                y_offset = (row * np.sqrt(3) * avg_radius_estimate) + avg_radius_estimate
                
                # Alternate x-offset for hexagonal packing
                x_start = avg_radius_estimate + (avg_radius_estimate if row % 2 == 1 else 0)
                
                col_idx = 0
                while True:
                    x_offset = x_start + col_idx * 2 * avg_radius_estimate
                    if x_offset > 1.0 - avg_radius_estimate: break # Stop if out of bounds
                    if len(initial_positions_list) < n_circles:
                        initial_positions_list.append([x_offset, y_offset])
                    else:
                        break
                    col_idx += 1
            
            initial_positions = np.array(initial_positions_list)[:n_circles]
            # Normalize and center if necessary (to fit 0.1-0.9 range)
            if initial_positions.size > 0:
                min_coords = np.min(initial_positions, axis=0)
                max_coords = np.max(initial_positions, axis=0)
                if np.any(max_coords - min_coords > 0): # Avoid division by zero
                    initial_positions = (initial_positions - min_coords) / (max_coords - min_coords) * 0.8 + 0.1
                else: # All points are identical, just center them
                    initial_positions = np.full_like(initial_positions, 0.5)
            else: # Fallback for empty array
                initial_positions = np.random.uniform(0.1, 0.9, (n_circles, 2))


            initial_radii = np.full(n_circles, 0.08)
            initial_positions += np.random.uniform(-0.01, 0.01, initial_positions.shape) # Small perturbation
            initial_positions = np.clip(initial_positions, 0.01, 0.99)
        
        else: # Strategy 4 & 5: More perturbed grid variations with random initial radii
            rows, cols = 5, 6
            x_coords = (np.arange(rows) + 0.5) / rows
            y_coords = (np.arange(cols) + 0.5) / cols
            xv, yv = np.meshgrid(x_coords, y_coords)
            initial_positions = np.vstack([xv.ravel(), yv.ravel()]).T[:n_circles]
            initial_positions += np.random.uniform(-0.03, 0.03, initial_positions.shape) # Even larger perturbation
            initial_positions = np.clip(initial_positions, 0.01, 0.99)
            initial_radii = np.full(n_circles, np.random.uniform(0.07, 0.09)) # Random initial radius

        x0 = np.hstack([initial_positions, initial_radii[:, np.newaxis]]).flatten()

        # Run the SLSQP optimizer.
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'disp': False, 'ftol': 1e-10} # Increased maxiter and ftol for deeper search
        )

        # Only consider successful optimizations for the best result
        if result.success:
            current_sum_radii = -result.fun
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_final_circles = result.x.reshape((n_circles, 3))
                # print(f"Run {i+1}: New best sum_radii = {best_sum_radii:.4f}")
        # else:
            # print(f"Run {i+1}: Optimization failed or did not converge: {result.message}. Sum radii: {-result.fun:.4f}")

    if best_final_circles is None:
        # Fallback if all optimizations fail to converge successfully.
        # This should ideally not happen with diverse initial guesses.
        print("Warning: All optimization runs failed to converge successfully. Returning initial guess from last run.")
        best_final_circles = x0.reshape((n_circles, 3)) # Fallback to the last initial guess

    return best_final_circles

# EVOLVE-BLOCK-END