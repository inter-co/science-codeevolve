# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize # New import for local optimization

def circle_packing32()->np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a Simulated Annealing metaheuristic to find a near-optimal configuration,
    followed by a local optimization step to ensure constraint satisfaction.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 32
    # Increased iterations for better exploration and convergence.
    N_ITER = 1000000 # Increased from 750000
    T_INITIAL = 0.5
    T_FINAL = 1e-10 # Lower T_FINAL for finer tuning
    
    # An exponential cooling schedule is calculated to transition from T_INITIAL to T_FINAL over N_ITER steps.
    COOLING_RATE = np.exp(np.log(T_FINAL / T_INITIAL) / N_ITER)

    # Fixed, higher penalty weights for constraint violations during SA.
    # The local optimizer will strictly enforce constraints, so SA mainly needs to find a good starting point.
    C_OVERLAP = 10000.0
    C_CONTAINMENT = 10000.0

    # Set a fixed seed for reproducibility of the stochastic process.
    np.random.seed(42)

    # Precompute upper-triangle indices for efficient pairwise calculations.
    # This avoids recomputing them in the performance-critical objective function.
    triu_indices = np.triu_indices(N_CIRCLES, k=1)

    def objective_function(circles: np.ndarray) -> float:
        """
        Calculates the cost of a given circle configuration.
        The cost is a combination of the negative sum of radii (to maximize it)
        and penalties for constraint violations (overlap and out-of-bounds).
        Radii are clamped to be non-negative for objective calculation.
        """
        radii = np.maximum(0, circles[:, 2]) # Clamp radii to be non-negative for calculation
        centers = circles[:, :2]

        # 1. Primary Objective: Maximize sum of radii. We minimize its negative.
        sum_radii = np.sum(radii)
        
        # 2. Containment Penalty: Circles must be fully within the unit square.
        x, y = centers.T
        p_contain = np.sum(np.maximum(0, radii - x)**2) + \
                    np.sum(np.maximum(0, (x + radii) - 1)**2) + \
                    np.sum(np.maximum(0, radii - y)**2) + \
                    np.sum(np.maximum(0, (y + radii) - 1)**2)

        # 3. Overlap Penalty: No two circles should overlap.
        p_overlap = 0
        if N_CIRCLES > 1:
            # Pairwise distances between circle centers
            dists = pdist(centers)
            # Corresponding sums of radii for each pair
            radii_sum_pairs = radii[triu_indices[0]] + radii[triu_indices[1]]
            # Overlap is the amount by which sum of radii exceeds distance
            overlaps = radii_sum_pairs - dists
            # Use a quadratic penalty for the magnitude of the overlap
            p_overlap = np.sum(np.maximum(0, overlaps)**2)

        # The total cost to be minimized by the annealer, using fixed penalty weights.
        cost = -sum_radii + C_OVERLAP * p_overlap + C_CONTAINMENT * p_contain
        return cost

    # --- Simulated Annealing Algorithm ---

    # 1. Initial State: Start with small circles randomly placed.
    current_state = np.random.rand(N_CIRCLES, 3)
    current_state[:, :2] *= 0.8 
    current_state[:, :2] += 0.1 # Place centers away from edges initially
    current_state[:, 2] *= 0.02 # Start with very small radii

    # Initial energy calculation, using fixed global penalty constants
    current_energy = objective_function(current_state)

    best_state = current_state.copy()
    best_energy = current_energy

    T = T_INITIAL

    # Main annealing loop
    for i in range(N_ITER):
        new_state = current_state.copy()
        
        # With a 20% probability, perform a radius swap move, which is a powerful non-local move.
        # Otherwise, perform the standard single-circle perturbation.
        if np.random.rand() < 0.2 and N_CIRCLES > 1:
            # --- Radius Swap Move ---
            idx1, idx2 = np.random.choice(N_CIRCLES, 2, replace=False)
            # Swap the radii
            new_state[idx1, 2], new_state[idx2, 2] = new_state[idx2, 2], new_state[idx1, 2]
        else:
            # --- Single Circle Perturbation Move ---
            # Select a random circle to modify
            idx = np.random.randint(0, N_CIRCLES)
            
            # Perturbation scales reverted to previously successful values
            # to balance exploration and fine-tuning.
            xy_scale = 0.1 * T + 0.0002 
            r_scale = 0.02 * T + 0.00005 

            # Perturb x, y, and r for the selected circle
            new_state[idx, 0] += np.random.normal(0, xy_scale)
            new_state[idx, 1] += np.random.normal(0, xy_scale)
            
            # Introduce a bias towards growing radii.
            radius_perturbation = np.random.normal(0, r_scale)
            if np.random.rand() < 0.7: # 70% chance to try growing radius
                radius_perturbation = abs(radius_perturbation) 
            new_state[idx, 2] += radius_perturbation
            
            # Clamp radius to be non-negative immediately after perturbation.
            new_state[idx, 2] = max(0, new_state[idx, 2])

        # 3. Evaluate the new state.
        new_energy = objective_function(new_state)

        # 4. Acceptance Criterion (Metropolis-Hastings)
        delta_E = new_energy - current_energy
        if delta_E < 0 or (T > T_FINAL and np.random.rand() < np.exp(-delta_E / T)):
            # Accept the new state
            current_state = new_state
            current_energy = new_energy

            # Update the best-so-far solution
            if current_energy < best_energy:
                best_state = current_state.copy()
                best_energy = current_energy
        
        # 5. Cool down the temperature.
        T *= COOLING_RATE

    # --- Post-Annealing Local Optimization ---
    # After SA, refine the best solution using a local optimizer (SLSQP)
    # This step is crucial for eliminating tiny overlaps and boundary violations
    # and ensuring the final configuration adheres strictly to constraints,
    # as the SA might tolerate very small violations at low temperatures.

    # 1. Define the objective function for local optimization.
    # We want to maximize the sum of radii, so we minimize its negative.
    def local_opt_objective_final(params_flat: np.ndarray) -> float:
        circles_opt = params_flat.reshape(N_CIRCLES, 3)
        radii = np.maximum(0, circles_opt[:, 2]) # Ensure non-negative radii
        return -np.sum(radii)

    # 2. Define bounds for x, y, r for each circle.
    # x and y coordinates must be within [0, 1]. Radii must be non-negative, up to 0.5.
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((0.0, 0.5)) # r_i (maximum possible radius is 0.5 for a circle to fit)

    # 3. Define constraints for containment and non-overlap.
    # These are inequality constraints g(x) >= 0.
    constraints = []

    # Helper function for containment constraints to avoid lambda capture issues
    def containment_constraint_func(params_flat, i_idx, type_idx, N):
        circles_opt = params_flat.reshape(N, 3)
        x_i, y_i, r_i = circles_opt[i_idx, 0], circles_opt[i_idx, 1], circles_opt[i_idx, 2]
        r_i_clamped = max(0, r_i) # Ensure non-negative radius for constraint evaluation

        if type_idx == 0: return x_i - r_i_clamped # x - r >= 0
        if type_idx == 1: return 1 - x_i - r_i_clamped # 1 - x - r >= 0
        if type_idx == 2: return y_i - r_i_clamped # y - r >= 0
        if type_idx == 3: return 1 - y_i - r_i_clamped # 1 - y - r >= 0
        raise ValueError("Invalid type_idx for containment constraint")

    # Containment constraints: r <= x <= 1-r and r <= y <= 1-r
    for i in range(N_CIRCLES):
        for type_idx in range(4): # 0:x-r, 1:1-x-r, 2:y-r, 3:1-y-r
            constraints.append({'type': 'ineq', 'fun': lambda params_flat, i_idx=i, t_idx=type_idx:
                                containment_constraint_func(params_flat, i_idx, t_idx, N_CIRCLES)})

    # Helper function for overlap constraint to avoid lambda capture issues
    def overlap_constraint_func(params_flat, i_idx, j_idx, N):
        """
        Calculates the overlap constraint using squared distances to avoid sqrt,
        which is computationally cheaper and can be more stable for optimizers.
        Constraint: dist_sq - (r_i + r_j)^2 >= 0
        """
        circles_opt = params_flat.reshape(N, 3)
        x_i, y_i, r_i = circles_opt[i_idx, 0], circles_opt[i_idx, 1], circles_opt[i_idx, 2]
        x_j, y_j, r_j = circles_opt[j_idx, 0], circles_opt[j_idx, 1], circles_opt[j_idx, 2]
        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
        r_i_clamped = max(0, r_i)
        r_j_clamped = max(0, r_j)
        return dist_sq - (r_i_clamped + r_j_clamped)**2

    # Non-overlap constraints: d_ij >= r_i + r_j
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            constraints.append({'type': 'ineq', 'fun': lambda params_flat, i_idx=i, j_idx=j:
                                overlap_constraint_func(params_flat, i_idx, j_idx, N_CIRCLES)})

    # Initial guess for the local optimizer is the best state from SA
    x0_local = best_state.flatten()

    # Perform local optimization using SLSQP
    # SLSQP is suitable for problems with bounds and inequality constraints.
    # Set a higher maxiter for potentially complex constraint landscape and lower ftol for higher precision.
    res = minimize(local_opt_objective_final, x0_local, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False}) # Increased maxiter to 3000, lowered ftol to 1e-9

    if res.success:
        optimized_circles = res.x.reshape(N_CIRCLES, 3)
        # Ensure radii are non-negative from the optimizer output.
        # The bounds and objective already encourage this, but a final clamp is safe.
        optimized_circles[:, 2] = np.maximum(0, optimized_circles[:, 2])
        return optimized_circles
    else:
        # If local optimization fails or does not converge, return the best state from SA.
        # This implies the SA solution might still have small violations.
        print(f"Warning: Local optimization failed: {res.message}. Returning best SA state after clamping radii.")
        best_state[:, 2] = np.maximum(0, best_state[:, 2]) # Ensure radii are non-negative
        return best_state

# EVOLVE-BLOCK-END
