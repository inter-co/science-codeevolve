# EVOLVE-BLOCK-START
# Core and external packages
import numpy as np
from scipy.optimize import minimize, linprog # Added linprog for Stage 1.5
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
from joblib import Parallel, delayed

# --- Problem Constants ---
N_CIRCLES = 21
BENCHMARK_SUM_RADII = 2.3658321334167627
PERIMETER = 4.0
PERIMETER_HALF = PERIMETER / 2.0 # width + height = 2
SEED = 42
EPSILON = 1e-9 # Stricter small constant for numerical stability (from Inspiration 3)
CONSTRAINT_TOLERANCE = 1e-9 # Tolerance for rigorous constraint checks (from Inspiration 2/3)
RECT_WIDTH_BOUNDS = (0.05, 1.95) # Wider bounds for rectangle width (from Inspiration 2/3)

# --- Global Helper Functions ---
def _extract_params(variables: np.ndarray):
    """
    Extracts container width, circle centers, and radii from the 1D optimization vector.
    Layout: [width, x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    This function is used for Stage 2 and Polishing.
    """
    container_width = variables[0]
    container_height = PERIMETER_HALF - container_width
    circles_data = variables[1:].reshape((N_CIRCLES, 3))
    centers = circles_data[:, :2]
    radii = circles_data[:, 2]
    return container_width, container_height, centers, radii

# --- Global Objective and Constraint Functions for Stage 2 and Polishing ---
def objective_function(variables: np.ndarray) -> float:
    """ Objective: Maximize sum of radii, so we minimize its negative. """
    _, _, _, radii = _extract_params(variables)
    return -np.sum(radii)

def _constraint_containment(variables: np.ndarray) -> np.ndarray:
    """ Constraint: Circles must be inside the rectangle (>= 0 when satisfied). """
    width, height, centers, radii = _extract_params(variables)
    return np.concatenate([
        centers[:, 0] - radii,              # x - r >= 0
        width - centers[:, 0] - radii,      # W - x - r >= 0
        centers[:, 1] - radii,              # y - r >= 0
        height - centers[:, 1] - radii      # H - y - r >= 0
    ])

def _constraint_non_overlap(variables: np.ndarray) -> np.ndarray:
    """ Constraint: Circles must not overlap (>= 0 when satisfied). Uses squared distances. """
    _, _, centers, radii = _extract_params(variables)
    if N_CIRCLES <= 1: return np.array([])
    
    center_distances_sq = pdist(centers, 'sqeuclidean')
    indices_rows, indices_cols = np.triu_indices(N_CIRCLES, k=1)
    radii_sums_sq = (radii[indices_rows] + radii[indices_cols])**2
    return center_distances_sq - radii_sums_sq # Squared distances vs squared sum of radii

def _constraint_positive_radii(variables: np.ndarray) -> np.ndarray:
    """ Constraint: Radii must be positive (>= 0 when satisfied). """
    _, _, _, radii = _extract_params(variables)
    return radii - EPSILON

# --- Validation and Visualization (kept for completeness, but not called in main flow) ---
def _validate_solution(circles: np.ndarray, width: float, height: float, tol: float = 1e-6) -> bool:
    """Rigorously checks if the final solution adheres to all constraints."""
    centers = circles[:, :2]
    radii = circles[:, 2]
    n_circles = len(circles)
    
    # Check 1: Positive radii
    if np.any(radii <= 0):
        print(f"Validation Error: Found non-positive radii.")
        return False
    
    # Check 2: Containment
    if np.any(centers[:, 0] < radii - tol) or np.any(centers[:, 0] > width - radii + tol):
        print(f"Validation Error: Circles exceed x-bounds.")
        return False
    if np.any(centers[:, 1] < radii - tol) or np.any(centers[:, 1] > height - radii + tol):
        print(f"Validation Error: Circles exceed y-bounds.")
        return False
        
    # Check 3: Non-overlap
    if n_circles > 1:
        dist_matrix = squareform(pdist(centers))
        radii_sum_matrix = np.add.outer(radii, radii)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-comparisons
        
        if np.any(dist_matrix < radii_sum_matrix - tol):
            print(f"Validation Error: Found overlapping circles.")
            return False
            
    print("Solution passed all validation checks.")
    return True

def _visualize_solution(circles: np.ndarray, width: float, height: float, title: str):
    """Generates a plot of the circle packing arrangement."""
    # Do not create plot in non-interactive environments (e.g. CI/CD)
    if 'DEBIAN_FRONTEND' in os.environ and os.environ['DEBIAN_FRONTEND'] == 'noninteractive':
        print("Skipping visualization in non-interactive environment.")
        return
        
    fig, ax = plt.subplots(1, dpi=120)
    ax.set_aspect('equal')

    # Draw rectangle
    rect = patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='r', facecolor='none', zorder=0)
    ax.add_patch(rect)

    # Draw circles
    for x, y, r in circles:
        circle = patches.Circle((x, y), r, facecolor=np.random.rand(3,), alpha=0.7, zorder=1)
        ax.add_patch(circle)

    ax.set_xlim(-0.1 * max(width, height), width + 0.1 * max(width, height))
    ax.set_ylim(-0.1 * max(width, height), height + 0.1 * max(width, height))
    ax.set_title(title)
    plt.grid(True)
    plt.show()


def circle_packing21()->np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 to maximize the sum of their radii.
    This implementation uses a parallelized, two-stage optimization approach with diversified structured initial guesses,
    LP radii pre-optimization, and a final polishing stage, inspired by high-performing constraint-based methods.
    """
    print("Starting optimization for 21 circles using parallel multi-stage SLSQP...")
    start_time_global = time.time()

    MASTER_SEED = SEED
    # Increased N_TRIALS to explore more of the solution space, leveraging more CPU cores.
    # Capped at 96 to balance performance and solution quality (from Inspiration 3).
    N_TRIALS = min(os.cpu_count() * 6 or 1, 128) # Increased N_TRIALS for broader exploration, preferring accuracy

    def _run_two_stage_optimization(seed: int):
        """
        Performs a full two-stage optimization (Stage 1: Uniform Radius, Stage 1.5: LP Radii, Stage 2: Variable Radii).
        Returns: (sum_radii, solution_vector, is_success, is_feasible)
        """
        local_rng = np.random.default_rng(seed)
        
        # --- Stage 1: Uniform Radius Optimization ---
        # Variable vector: [r, w, c1_x, c1_y, c2_x, c2_y, ...]
        def _extract_s1(v):
            r, w = v[0], v[1]
            h = PERIMETER_HALF - w
            c = v[2:].reshape((N_CIRCLES, 2))
            return r, w, h, c

        def _obj_s1(v): return -v[0] # Maximize radius r

        def _con_contain_s1(v):
            r, w, h, c = _extract_s1(v)
            return np.concatenate([c[:, 0] - r, w - c[:, 0] - r, c[:, 1] - r, h - c[:, 1] - r])

        def _con_overlap_s1(v):
            r, _, _, c = _extract_s1(v)
            if N_CIRCLES <= 1: return np.array([])
            dist_sq = pdist(c, 'sqeuclidean')
            return dist_sq - (2 * r)**2

        # Diversified initial width selection for Stage 1 (from Inspiration 2)
        initial_widths_options = [
            1.225, 1.235, 1.215, 1.245, # Tightly concentrated around known optimal W for N=21
            1.0, 1.618, 0.823, 1.3, 0.75 # Broader exploration of common/golden ratio/well-known aspect ratios
        ]
        w0 = initial_widths_options[seed % len(initial_widths_options)]
        w0 += local_rng.uniform(-0.05, 0.05) # Add jitter to width
        w0 = np.clip(w0, RECT_WIDTH_BOUNDS[0], RECT_WIDTH_BOUNDS[1])
        h0 = PERIMETER_HALF - w0
        
        # Diversify initial grid structure (3x7 or 7x3) based on seed.
        n_rows, n_cols = (3, 7) if seed % 2 == 0 else (7, 3)
        
        # Calculate grid spacing for adaptive jitter
        x_spacing = w0 / (n_cols + 1)
        y_spacing = h0 / (n_rows + 1)

        # Introduce random perturbation to linspace start/end points for more diverse initial grids (from Inspiration 2)
        x_start_factor = x_spacing + local_rng.uniform(-x_spacing * 0.1, x_spacing * 0.1)
        x_end_factor = w0 - x_spacing + local_rng.uniform(-x_spacing * 0.1, x_spacing * 0.1)
        y_start_factor = y_spacing + local_rng.uniform(-y_spacing * 0.1, y_spacing * 0.1)
        y_end_factor = h0 - y_spacing + local_rng.uniform(-y_spacing * 0.1, y_spacing * 0.1)

        grid_x = np.linspace(x_start_factor, x_end_factor, n_cols)
        grid_y = np.linspace(y_start_factor, y_end_factor, n_rows)
        
        xx, yy = np.meshgrid(grid_x, grid_y)
        c0 = np.vstack([xx.ravel(), yy.ravel()]).T[:N_CIRCLES] # Ensure exactly N_CIRCLES centers

        # Add adaptive jitter to centers, proportional to grid spacing (from Inspiration 2)
        jitter_mag_x = x_spacing * 0.2
        jitter_mag_y = y_spacing * 0.2
        c0 += local_rng.uniform(-1, 1, size=c0.shape) * np.array([jitter_mag_x, jitter_mag_y])
        
        # Clip centers to ensure they are within reasonable bounds, considering radii
        c0[:, 0] = np.clip(c0[:, 0], EPSILON, w0 - EPSILON)
        c0[:, 1] = np.clip(c0[:, 1], EPSILON, h0 - EPSILON)

        r0 = EPSILON # Fixed initial radius for Stage 1.
        
        # Ensure initial centers are within bounds, considering their newly generated initial radii (from Inspiration 1)
        for k in range(N_CIRCLES):
            c0[k, 0] = np.clip(c0[k, 0], r0, w0 - r0)
            c0[k, 1] = np.clip(c0[k, 1], r0, h0 - r0)

        x0_s1 = np.concatenate([np.array([r0, w0]), c0.flatten()])
        # Bounds for r, w, then x,y for each circle.
        # Harmonized x,y bounds to be consistent with RECT_WIDTH_BOUNDS for the container.
        bounds_s1 = [(EPSILON, PERIMETER_HALF / 2.0)] + [RECT_WIDTH_BOUNDS] + \
                    [(0.0, RECT_WIDTH_BOUNDS[1])] * (N_CIRCLES * 2)
        cons_s1 = [{'type': 'ineq', 'fun': _con_contain_s1}, {'type': 'ineq', 'fun': _con_overlap_s1}]
        
        res_s1 = minimize(_obj_s1, x0_s1, method='SLSQP', bounds=bounds_s1, constraints=cons_s1,
                          options={'maxiter': 3000, 'disp': False, 'ftol': 1e-9})

        if not res_s1.success:
            return -np.inf, None, False, False # sum_radii, solution_vector, is_success, is_feasible

        # --- Stage 1.5: LP Radii Pre-optimization for a superior warm start (from Inspiration 3) ---
        r1_uniform, w1, h1, c1 = _extract_s1(res_s1.x) # Use results from Stage 1

        c_lp = -np.ones(N_CIRCLES) # Objective: maximize sum(r)

        bounds_r = [] # Bounds for each radius based on containment within the rectangle
        for i in range(N_CIRCLES):
            max_r = min(c1[i, 0], w1 - c1[i, 0], c1[i, 1], h1 - c1[i, 1])
            bounds_r.append((EPSILON, max_r if max_r > EPSILON else EPSILON))

        # Overlap constraints: r_i + r_j <= dist(c_i, c_j)
        distances = squareform(pdist(c1)) # Use squareform to get N x N distance matrix
        indices_rows, indices_cols = np.triu_indices(N_CIRCLES, k=1)
        num_overlap_constraints = len(indices_rows)
        A_lp = np.zeros((num_overlap_constraints, N_CIRCLES))
        b_lp = np.zeros(num_overlap_constraints)

        for i, (row, col) in enumerate(zip(indices_rows, indices_cols)):
            A_lp[i, row] = 1
            A_lp[i, col] = 1
            b_lp[i] = distances[row, col]
        
        lp_res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=bounds_r, method='highs')

        if lp_res.success and lp_res.fun is not None:
            r_initial_s2 = lp_res.x
        else:
            # Fallback to uniform radius from Stage 1 if LP solver fails
            r_initial_s2 = np.full(N_CIRCLES, r1_uniform)

        # Assemble the initial guess vector for Stage 2
        x0_s2 = np.concatenate([np.array([w1]), np.hstack([c1, r_initial_s2.reshape(-1, 1)]).flatten()])
        
        # --- Stage 2: Variable Radii Optimization (Warm Start) ---
        bounds_s2 = [RECT_WIDTH_BOUNDS] # Rectangle width bounds
        for _ in range(N_CIRCLES):
            # x, y bounds looser, containment constraints will tighten
            # r bounds from EPSILON to max possible radius (half of min dim, which can be up to 1.0)
            bounds_s2.extend([(0.0, RECT_WIDTH_BOUNDS[1]), (0.0, RECT_WIDTH_BOUNDS[1]), (EPSILON, 1.0)])
        
        # Use globally defined constraints for Stage 2
        cons_s2 = [{'type': 'ineq', 'fun': _constraint_containment},
                   {'type': 'ineq', 'fun': _constraint_non_overlap},
                   {'type': 'ineq', 'fun': _constraint_positive_radii}]

        res_s2 = minimize(objective_function, x0_s2, method='SLSQP', bounds=bounds_s2, constraints=cons_s2,
                          options={'maxiter': 12000, 'disp': False, 'ftol': 1e-10})
        
        # Rigorously check feasibility of the Stage 2 result (from Inspirations 2 & 3)
        is_feasible_s2 = False
        if res_s2.success:
            containment_ok = np.all(_constraint_containment(res_s2.x) >= -CONSTRAINT_TOLERANCE)
            overlap_ok = np.all(_constraint_non_overlap(res_s2.x) >= -CONSTRAINT_TOLERANCE)
            positive_radii_ok = np.all(_constraint_positive_radii(res_s2.x) >= -CONSTRAINT_TOLERANCE)
            is_feasible_s2 = containment_ok and overlap_ok and positive_radii_ok

        # Return relevant info for aggregation, including feasibility status
        return -res_s2.fun, res_s2.x, res_s2.success, is_feasible_s2

    # --- Main Multi-start Logic ---
    rng = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(low=0, high=2**31, size=N_TRIALS)

    print(f"Starting {N_TRIALS} parallel multi-stage optimization trials...")
    results = Parallel(n_jobs=-1)(delayed(_run_two_stage_optimization)(seed) for seed in seeds)
    print(f"\nGlobal search finished in {time.time() - start_time_global:.2f} seconds.")

    best_sum_radii = -np.inf
    best_solution_vector = None
    
    # Aggregate results: find the best feasible solution
    feasible_results = []
    for sum_r, sol_vec, success, feasible in results:
        if success and feasible:
            feasible_results.append((sum_r, sol_vec))
            if sum_r > best_sum_radii:
                best_sum_radii = sum_r
                best_solution_vector = sol_vec
    
    if best_solution_vector is None:
        print("\nFATAL: No feasible solution found across all runs. Returning an array of zeros.")
        return np.zeros((N_CIRCLES, 3))

    # Extract initial best parameters for reporting
    initial_best_width, initial_best_height, _, _ = _extract_params(best_solution_vector)
    print(f"Initial best sum_radii from multi-start: {best_sum_radii:.12f}")
    print(f"Initial optimal container dimensions: Width={initial_best_width:.6f}, Height={initial_best_height:.6f}")

    # --- Stage 3: Final High-Precision Refinement (Polishing) (from Inspirations 2 & 3) ---
    print(f"\nStarting high-precision refinement on best solution (sum_radii = {best_sum_radii:.12f})...")
    start_time_polish = time.time()
    
    x0_polish = best_solution_vector # Use the best raw vector found in the global search as the starting point

    # Define bounds for polishing (same as Stage 2)
    bounds_polish = [RECT_WIDTH_BOUNDS]
    for _ in range(N_CIRCLES):
        bounds_polish.extend([(0.0, RECT_WIDTH_BOUNDS[1]), (0.0, RECT_WIDTH_BOUNDS[1]), (EPSILON, 1.0)])
    
    # Use globally defined constraints for polishing
    cons_polish = [{'type': 'ineq', 'fun': _constraint_containment},
                   {'type': 'ineq', 'fun': _constraint_non_overlap},
                   {'type': 'ineq', 'fun': _constraint_positive_radii}]

    polish_result = minimize(
        objective_function, x0_polish, method='SLSQP', bounds=bounds_polish, constraints=cons_polish,
        options={'maxiter': 75000, 'disp': False, 'ftol': 1e-14} # Extremely strict tolerance for polishing
    )

    polished_sum_radii = best_sum_radii # Default to previous best
    final_circles_solution = None

    if polish_result.success:
        current_polished_sum_radii = -polish_result.fun
        # Rigorously check feasibility of the polished solution
        is_feasible_polish = (
            np.all(_constraint_containment(polish_result.x) >= -CONSTRAINT_TOLERANCE) and
            np.all(_constraint_non_overlap(polish_result.x) >= -CONSTRAINT_TOLERANCE) and
            np.all(_constraint_positive_radii(polish_result.x) >= -CONSTRAINT_TOLERANCE)
        )
        if is_feasible_polish and current_polished_sum_radii > best_sum_radii:
            polished_sum_radii = current_polished_sum_radii
            width, height, centers, radii = _extract_params(polish_result.x)
            final_circles_solution = np.hstack((centers, radii.reshape(-1, 1)))
            print(f"Refinement successful! Improved sum_radii from {best_sum_radii:.12f} to {polished_sum_radii:.12f}")
        else:
            print("Refinement did not yield a better feasible solution or failed feasibility check. Using best pre-polished solution.")
            width, height, centers, radii = _extract_params(best_solution_vector)
            final_circles_solution = np.hstack((centers, radii.reshape(-1, 1)))
    else:
        print(f"Refinement failed: {polish_result.message}. Using best pre-polished solution.")
        width, height, centers, radii = _extract_params(best_solution_vector)
        final_circles_solution = np.hstack((centers, radii.reshape(-1, 1)))
    
    end_time_polish = time.time()
    print(f"Refinement finished in {end_time_polish - start_time_polish:.2f} seconds.")

    # Final clipping to ensure strict adherence to non-negative radii (from Inspirations)
    final_circles_solution[:, 2] = np.maximum(EPSILON, final_circles_solution[:, 2])

    # Post-optimization clipping for centers to ensure strict containment (from Inspiration 1 & 3)
    final_optimal_vector = polish_result.x if (polish_result.success and is_feasible_polish and polished_sum_radii > best_sum_radii) else best_solution_vector
    final_width, final_height, _, _ = _extract_params(final_optimal_vector)

    for i in range(N_CIRCLES):
        r_i = final_circles_solution[i, 2] # Use the (potentially clipped) final radius
        final_circles_solution[i, 0] = np.clip(final_circles_solution[i, 0], r_i, final_width - r_i)
        final_circles_solution[i, 1] = np.clip(final_circles_solution[i, 1], r_i, final_height - r_i)

    print(f"\n--- Overall Best Result ---")
    print(f"Total Sum of Radii: {polished_sum_radii:.12f}")
    
    print(f"Optimal Container Dimensions: Width={final_width:.6f}, Height={final_height:.6f}")
    print(f"Benchmark Ratio (vs AlphaEvolve {BENCHMARK_SUM_RADII:.12f}): {polished_sum_radii / BENCHMARK_SUM_RADII:.12f}")

    # The _validate_solution and _visualize_solution calls are now external to the constructor function.
    # They can be called from if __name__ == '__main__': if needed.
    # _validate_solution(final_circles_solution, final_width, final_height, tol=CONSTRAINT_TOLERANCE)
    # title = f"21 Circles Packing | Sum Radii: {polished_sum_radii:.6f}"
    # _visualize_solution(final_circles_solution, final_width, final_height, title)

    return final_circles_solution

# EVOLVE-BLOCK-END

if __name__ == '__main__':
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
