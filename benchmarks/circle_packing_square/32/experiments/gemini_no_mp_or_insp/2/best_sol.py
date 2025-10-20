# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import njit, float64
import time # For eval_time measurement
from joblib import Parallel, delayed # For parallel local searches

# Constants
N_CIRCLES = 32
# Increased penalty coefficients to enforce constraints more strictly
PENALTY_COEFF_BOUNDARY = 1e6 # Increased penalty coefficients to enforce constraints more strictly
PENALTY_COEFF_OVERLAP = 1e6 # Increased penalty coefficients to enforce constraints more strictly 
MIN_RADIUS = 1e-6 # Ensure radii are positive
MAX_RADIUS_GUESS = 0.5 - MIN_RADIUS # Allow larger radii, up to 0.5 (theoretical max for a single circle)
# Increased iterations for more thorough search
GLOBAL_OPTIMIZER_MAXITER = 1500 # Increased from 1000 for more global exploration
LOCAL_OPTIMIZER_MAXITER = 1500 # Increased from 1000 for more local refinement
SEED = 42 # For reproducibility

# --- Numba-jitted helper functions for speed ---

@njit(float64[:](float64[:])) # Returns a 2-element array [boundary_penalty, overlap_penalty]
def calculate_penalties_numba(params):
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    boundary_penalty = 0.0
    for i in range(n):
        # r <= x <= 1-r  =>  x-r >= 0, 1-x-r >= 0
        boundary_penalty += max(0.0, r[i] - x[i])**2 # Penalize squared violation
        boundary_penalty += max(0.0, x[i] + r[i] - 1.0)**2
        boundary_penalty += max(0.0, r[i] - y[i])**2
        boundary_penalty += max(0.0, y[i] + r[i] - 1.0)**2
        # Ensure radius is non-negative
        boundary_penalty += max(0.0, -r[i])**2

    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            # Penalize squared violation for a smoother objective function landscape (C1 continuous)
            overlap_penalty += max(0.0, min_dist_sq - dist_sq)**2 
    return np.array([boundary_penalty, overlap_penalty])

# Numba-jitted constraint functions for SLSQP for high performance
@njit(float64[:](float64[:]))
def slsqp_boundary_constraints_numba(params):
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    # Constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    constraints = np.empty(4 * n, dtype=np.float64)
    for i in range(n):
        constraints[i] = x[i] - r[i]
        constraints[n + i] = 1.0 - x[i] - r[i]
        constraints[2 * n + i] = y[i] - r[i]
        constraints[3 * n + i] = 1.0 - y[i] - r[i]
    return constraints

@njit(float64[:](float64[:]))
def slsqp_overlap_constraints_numba(params):
    n = len(params) // 3
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    num_constraints = n * (n - 1) // 2
    constraints = np.empty(num_constraints, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            radii_sum_sq = (r[i] + r[j])**2
            constraints[k] = dist_sq - radii_sum_sq
            k += 1
    return constraints

# --- Physics-Based Refinement (Post-processing) ---

@njit
def _calculate_forces_numba(pos, radii, k_repel=1.0):
    """Calculates repulsive forces for overlaps and boundary violations."""
    n = len(pos)
    forces = np.zeros_like(pos)
    
    # Overlap forces
    for i in range(n):
        for j in range(i + 1, n):
            vec = pos[j] - pos[i]
            dist_sq = vec[0]**2 + vec[1]**2
            radii_sum = radii[i] + radii[j]
            
            if dist_sq < radii_sum**2 - 1e-12:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                overlap = radii_sum - dist
                force_magnitude = k_repel * overlap
                force_vec = (vec / dist) * force_magnitude
                forces[i] -= force_vec
                forces[j] += force_vec

    # Boundary forces
    for i in range(n):
        if pos[i, 0] < radii[i]: forces[i, 0] += k_repel * (radii[i] - pos[i, 0])
        if pos[i, 0] > 1.0 - radii[i]: forces[i, 0] -= k_repel * (pos[i, 0] - (1.0 - radii[i]))
        if pos[i, 1] < radii[i]: forces[i, 1] += k_repel * (radii[i] - pos[i, 1])
        if pos[i, 1] > 1.0 - radii[i]: forces[i, 1] -= k_repel * (pos[i, 1] - (1.0 - radii[i]))
            
    return forces

@njit
def _relax_positions_numba(pos, radii, relax_iter=200, dt=0.01, damping=0.95):
    """Adjusts circle positions via velocity-based physics simulation to minimize overlaps."""
    n = len(pos)
    velocity = np.zeros_like(pos)
    for _ in range(relax_iter):
        forces = _calculate_forces_numba(pos, radii, k_repel=1.0)
        
        if np.sum(forces**2) < 1e-12: # Converged if forces are negligible
            break
        
        velocity += forces * dt
        velocity *= damping # Apply damping to bleed off energy
        pos += velocity * dt
        
        # Numba-compatible clip to keep circles within the unit square
        for i in range(n):
            pos[i, 0] = min(max(pos[i, 0], 0.0), 1.0)
            pos[i, 1] = min(max(pos[i, 1], 0.0), 1.0)
            
    return pos

@njit
def _grow_radii_simultaneous_numba(pos, radii, grow_iter, grow_rate): # Removed default values, passed explicitly
    """Grows all circle radii simultaneously based on available space."""
    n = len(pos)
    current_radii = radii.copy()
    for _ in range(grow_iter):
        max_radii = np.empty_like(current_radii)
        
        for i in range(n):
            xi, yi = pos[i]
            max_r_boundary = min(xi, 1.0 - xi, yi, 1.0 - yi)
            
            min_dist_to_neighbor_edge = np.inf
            for j in range(n):
                if i == j: continue
                xj, yj = pos[j]
                rj = current_radii[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist_to_neighbor_edge = min(min_dist_to_neighbor_edge, dist - rj)
            
            max_radii[i] = max(MIN_RADIUS, min(max_r_boundary, min_dist_to_neighbor_edge))

        growth_potential = max_radii - current_radii
        
        if np.all(growth_potential < 1e-12): # Converged if no growth is possible
            break
        
        # Grow all circles by a fraction of their potential to avoid overshooting
        current_radii += np.maximum(0.0, growth_potential) * grow_rate
        
    return current_radii

def post_process_solution(circles, shrink_iter=5, relax_grow_cycles=7, relax_iter=200, grow_iter=100): # Increased cycles to 7
    """Refines a circle configuration using an improved shrink, relax, and grow strategy."""
    n = len(circles)
    
    # Part 1: Aggressive Shrink to guarantee a valid starting point
    for _ in range(shrink_iter):
        violations_found = False
        # Boundary shrink
        for i in range(n):
            x, y, r = circles[i]
            r_max_boundary = min(x, 1-x, y, 1-y)
            if r > r_max_boundary + 1e-12:
                circles[i, 2] = r_max_boundary * 0.9999
                violations_found = True
        # Overlap shrink
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]; xj, yj, rj = circles[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                radii_sum = ri + rj
                if dist_sq < radii_sum**2 - 1e-12:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                    overlap = radii_sum - dist
                    if radii_sum > 1e-9:
                       # Shrink proportionally to radius
                       circles[i, 2] -= overlap * (ri / radii_sum) * 0.51
                       circles[j, 2] -= overlap * (rj / radii_sum) * 0.51
                    violations_found = True
        if not violations_found: break

    # Part 2: Iterative Physics-based Relaxation and Simultaneous Growth
    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()

    for cycle in range(relax_grow_cycles):
        # Relax positions using velocity-based simulation
        pos = _relax_positions_numba(pos, radii, relax_iter=relax_iter)

        # Grow radii using simultaneous growth model with an adaptive grow_rate
        current_grow_rate = 0.2 if cycle < relax_grow_cycles // 2 else 0.05 # Higher grow rate for initial cycles, then lower for refinement
        new_radii = _grow_radii_simultaneous_numba(pos, radii, grow_iter=grow_iter, grow_rate=current_grow_rate)
        
        # Check for convergence across a full relax-grow cycle
        if np.sum(new_radii - radii) < 1e-9:
             break
        radii = new_radii

    circles[:, :2] = pos
    circles[:, 2] = radii
    return circles


# Function to generate a smart initial population for DE
def generate_initial_population(n, popsize, seed):
    rng = np.random.default_rng(seed)
    population = np.zeros((popsize, n * 3))
    
    # Helper for generating hexagonal grid points
    def _generate_hex_grid_points(n_circles, square_size=1.0, jitter_factor=0.1, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Estimate base radius for a roughly hexagonal packing
        # Refined hexagonal grid generation for better initial guesses
        
        # Estimate base radius for a roughly hexagonal packing, adjusted for finite boundaries.
        # The factor 0.98 (increased from 0.95) allows for slightly larger initial radii,
        # providing more room for DE to shrink if needed, or grow if possible.
        base_r = max(MIN_RADIUS, min(np.sqrt(1.0 / (n_circles * 2 * np.sqrt(3))) * 0.98, MAX_RADIUS_GUESS * 0.7))

        # Calculate grid dimensions based on the estimated base_r
        # Increased buffer to ensure enough points are generated even if base_r is large
        grid_cols_needed = int(1.0 / (2 * base_r)) + 3 
        grid_rows_needed = int(1.0 / (np.sqrt(3) * base_r)) + 3
        
        points = []
        # Initial radius guess for hex pattern, with some variability.
        # Start closer to base_r to be more aggressive.
        r_initial_guess = base_r * rng.uniform(0.8, 1.0) 

        for i in range(grid_rows_needed):
            y_center = i * np.sqrt(3) * base_r + base_r 
            
            x_offset_row = 0.0 if i % 2 == 0 else base_r
            
            for j in range(grid_cols_needed):
                x_center = x_offset_row + j * 2 * base_r + base_r
                
                # Add jitter, now relative to base_r for consistency.
                # Slightly increased jitter for more diversity
                x_pos = x_center + rng.uniform(-0.07 * base_r, 0.07 * base_r)
                y_pos = y_center + rng.uniform(-0.07 * base_r, 0.07 * base_r)
                
                # Clip to unit square, ensuring space for MIN_RADIUS
                x_pos = np.clip(x_pos, MIN_RADIUS, 1 - MIN_RADIUS)
                y_pos = np.clip(y_pos, MIN_RADIUS, 1 - MIN_RADIUS)
                
                points.append([x_pos, y_pos])
                if len(points) == n_circles:
                    break
            if len(points) == n_circles:
                break
                
        # Fallback if not enough points generated (should be rare with buffer)
        while len(points) < n_circles:
            points.append([rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS), rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)])
            
        return np.array(points[:n_circles]), r_initial_guess

    # Strategy distribution for initial population
    strategy_weights = [0.30, 0.30, 0.15, 0.15, 0.10] # Square, Hex, Anchors, Bimodal, Random
    strategy_counts = [int(w * popsize) for w in strategy_weights]
    # Distribute any remaining count due to integer truncation
    remaining = popsize - sum(strategy_counts)
    for i in range(remaining):
        strategy_counts[i] += 1

    current_idx = 0

    # Strategy 1: Jittered Square Grid
    for k in range(strategy_counts[0]):
        grid_size = int(np.ceil(np.sqrt(n)))
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                points.append([(i + 0.5) / grid_size, (j + 0.5) / grid_size])
        
        base_points = np.array(points[:n])
        jitter = rng.uniform(-0.25 / grid_size, 0.25 / grid_size, size=(n, 2))
        centers = base_points + jitter
        
        initial_r = rng.uniform(MIN_RADIUS, MAX_RADIUS_GUESS / (grid_size * 1.5), size=n) # Allow slightly larger radii
        
        params = np.zeros(n * 3)
        params[0::3] = np.clip(centers[:, 0], MIN_RADIUS, 1 - MIN_RADIUS)
        params[1::3] = np.clip(centers[:, 1], MIN_RADIUS, 1 - MIN_RADIUS)
        params[2::3] = np.clip(initial_r, MIN_RADIUS, MAX_RADIUS_GUESS)
        population[current_idx + k] = params
    current_idx += strategy_counts[0]

    # Strategy 2: Jittered Hexagonal Grid
    for k in range(strategy_counts[1]):
        hex_points, r_hex_base = _generate_hex_grid_points(n, rng=rng)
        jitter = rng.uniform(-0.02, 0.02, size=(n, 2)) # Smaller jitter for hex patterns
        centers = hex_points + jitter
        
        initial_r = rng.uniform(MIN_RADIUS, r_hex_base * 1.2, size=n) # Allow slightly larger than base hex radius
        
        params = np.zeros(n * 3)
        params[0::3] = np.clip(centers[:, 0], MIN_RADIUS, 1 - MIN_RADIUS)
        params[1::3] = np.clip(centers[:, 1], MIN_RADIUS, 1 - MIN_RADIUS)
        params[2::3] = np.clip(initial_r, MIN_RADIUS, MAX_RADIUS_GUESS)
        population[current_idx + k] = params
    current_idx += strategy_counts[1]

    # Strategy 3: Corner/Edge Anchors + Random Fill
    num_anchors = min(n, 4) # Place up to 4 large circles at corners
    for k in range(strategy_counts[2]):
        params = np.zeros(n * 3)
        
        # Place anchors (e.g., corners) with larger radii
        anchor_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        anchor_radii = rng.uniform(0.1, MAX_RADIUS_GUESS * 0.8, size=num_anchors) # Larger radii for anchors
        
        for i in range(num_anchors):
            r_val = np.clip(anchor_radii[i], MIN_RADIUS, MAX_RADIUS_GUESS)
            # Position center 'r' units from the boundary
            params[i*3+0] = np.clip(anchor_coords[i, 0] + r_val, MIN_RADIUS, 1 - MIN_RADIUS) if anchor_coords[i,0] == 0 else np.clip(anchor_coords[i,0] - r_val, MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+1] = np.clip(anchor_coords[i, 1] + r_val, MIN_RADIUS, 1 - MIN_RADIUS) if anchor_coords[i,1] == 0 else np.clip(anchor_coords[i,1] - r_val, MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+2] = r_val
        
        # Fill remaining circles randomly with smaller radii
        for i in range(num_anchors, n):
            params[i*3+0] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+1] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+2] = rng.uniform(MIN_RADIUS, MAX_RADIUS_GUESS * 0.5) # Smaller radii for fillers
        population[current_idx + k] = params
    current_idx += strategy_counts[2]

    # Strategy 4: Bimodal (few large, many small)
    for k in range(strategy_counts[3]):
        params = np.zeros(n * 3)
        num_large = rng.integers(4, 9) # e.g., 4-8 large circles

        # Large circles
        for i in range(num_large):
            params[i*3+0] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+1] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+2] = rng.uniform(0.1, 0.22) # Large radii
        
        # Small circles
        for i in range(num_large, n):
            params[i*3+0] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+1] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS)
            params[i*3+2] = rng.uniform(0.02, 0.08) # Small filler radii
        population[current_idx + k] = params
    current_idx += strategy_counts[3]

    # Strategy 5: Pure Random
    for k in range(strategy_counts[4]):
        params = np.zeros(n * 3)
        params[0::3] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS, size=n)
        params[1::3] = rng.uniform(MIN_RADIUS, 1 - MIN_RADIUS, size=n)
        params[2::3] = rng.uniform(MIN_RADIUS, MAX_RADIUS_GUESS, size=n)
        population[current_idx + k] = params
    current_idx += strategy_counts[4]

    return population

# --- Objective and Validation Functions ---

def objective_penalized(params):
    radii = params[2::3]
    sum_radii = np.sum(radii)
    penalties = calculate_penalties_numba(params)
    boundary_penalty = penalties[0]
    overlap_penalty = penalties[1]
    return -sum_radii + PENALTY_COEFF_BOUNDARY * boundary_penalty + PENALTY_COEFF_OVERLAP * overlap_penalty

def objective_slsqp(params):
    return -np.sum(params[2::3])

def validate_circles(circles):
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    for i in range(n):
        if not (r[i] <= x[i] <= 1 - r[i] and r[i] <= y[i] <= 1 - r[i]):
            print(f"Validation Error: Boundary violation for circle {i}: x={x[i]:.4f}, y={y[i]:.4f}, r={r[i]:.4f}")
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist_sq = (r[i] + r[j])**2
            if dist_sq < min_dist_sq - 1e-9:
                print(f"Validation Error: Overlap violation for circles {i} and {j}: dist_sq={dist_sq:.6f}, min_dist_sq={min_dist_sq:.6f}")
                return False
    return True

# --- Main Constructor Function ---

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    
    # --- Phase 1: Global Optimization with Differential Evolution ---
    # Use a smart initial population based on a jittered grid
    # Determine an appropriate popsize based on number of parameters
    # N_CIRCLES * 3 parameters (x, y, r) = 32 * 3 = 96 parameters
    # A popsize of 3-5x the number of parameters is often recommended for DE.
    # We choose 300 to balance exploration depth with reasonable evaluation time.
    de_popsize = 300 # Increased popsize from 100 to 300 for better global exploration

    initial_population = generate_initial_population(n, popsize=de_popsize, seed=SEED)

    # Define bounds for x, y, r. Tighter radius bound helps optimizer.
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (MIN_RADIUS, MAX_RADIUS_GUESS)])

    start_time_global = time.time()
    print("Starting global optimization with Differential Evolution...")
    result_de = differential_evolution(
        objective_penalized,
        bounds,
        init=initial_population,
        maxiter=GLOBAL_OPTIMIZER_MAXITER,
        popsize=de_popsize, # Use the calculated popsize
        workers=-1, # Use all available CPU cores
        disp=False,
        seed=SEED,
        tol=1e-5, # Tighter tolerance
        polish=False # We use a more powerful local optimizer next
    )
    end_time_global = time.time()
    print(f"Differential Evolution finished in {end_time_global - start_time_global:.2f} seconds.")
    # Calculate approximate sum_radii from DE result
    approx_sum_radii = np.sum(result_de.x[2::3])
    print(f"DE best objective: {result_de.fun:.4f}, approx sum_radii: {approx_sum_radii:.4f}")

    # --- Phase 2: Multi-Start Local Refinement with SLSQP ---
    # We take the top candidates from DE and refine them in parallel
    
    # Evaluate fitness of the final DE population
    population = result_de.population
    pop_fitness = np.array([objective_penalized(ind) for ind in population])
    
    # Get indices of the top N candidates
    N_TOP_CANDIDATES = 20 # Increased from 10 to 20 for wider local search exploration
    top_indices = np.argsort(pop_fitness)[:N_TOP_CANDIDATES]
    top_candidates = population[top_indices]

    slsqp_bounds = bounds
    slsqp_constraints = [
        {'type': 'ineq', 'fun': slsqp_boundary_constraints_numba},
        {'type': 'ineq', 'fun': slsqp_overlap_constraints_numba}
    ]

    def run_slsqp(x0):
        # Helper function for parallel execution
        return minimize(
            objective_slsqp,
            x0,
            method='SLSQP',
            bounds=slsqp_bounds,
            constraints=slsqp_constraints,
            options={'maxiter': LOCAL_OPTIMIZER_MAXITER, 'ftol': 1e-9, 'disp': False}
        )

    print(f"Starting local refinement for top {N_TOP_CANDIDATES} DE candidates in parallel...")
    start_time_local = time.time()
    
    # Use joblib to run SLSQP in parallel
    results_slsqp = Parallel(n_jobs=-1)(delayed(run_slsqp)(x0) for x0 in top_candidates)
    
    end_time_local = time.time()
    print(f"Parallel SLSQP finished in {end_time_local - start_time_local:.2f} seconds.")

    # Find the best result among all SLSQP runs
    best_result = None
    best_fun = np.inf
    for res in results_slsqp:
        if res.success and res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    if best_result is None:
        # Fallback if all SLSQP runs failed, which is unlikely
        print("Warning: All SLSQP runs failed. Using the best DE result directly.")
        best_result = result_de
    
    print(f"Best SLSQP objective: {best_result.fun:.6f}, sum_radii: {-best_result.fun:.6f}")
    print(f"Best SLSQP success: {best_result.success}, message: {best_result.message}")

    final_params = best_result.x
    circles = final_params.reshape(n, 3)

    # --- Phase 3: Post-processing with Improved Physics-Based Refinement ---
    print("Starting post-processing: shrink, relax, and grow...")
    initial_sum_radii = np.sum(circles[:, 2])
    # Use the globally defined, improved post-processing function.
    # This new version uses a more robust velocity-based relaxation and simultaneous growth.
    circles = post_process_solution(circles.copy())
    final_sum_radii = np.sum(circles[:, 2])

    if not validate_circles(circles):
         print("Warning: Solution still violates constraints after post-processing.")
    else:
        print(f"Post-processing complete. Sum radii changed from {initial_sum_radii:.6f} to {final_sum_radii:.6f}.")

    return circles

# EVOLVE-BLOCK-END
