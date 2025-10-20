# EVOLVE-BLOCK-START
import numpy as np
# Import necessary libraries for optimization and distance calculation
from scipy.optimize import differential_evolution, minimize # Added minimize for local refinement
from scipy.spatial.distance import pdist

# Helper function to generate points for a regular N-gon with perturbation
def _generate_regular_polygon_points(n_sides, center, radius, perturbation_scale, seed):
    np.random.seed(seed)
    points = np.zeros((n_sides, 2))
    # Start angle can be randomized for diversity in restarts
    start_angle = np.random.uniform(0, 2 * np.pi / n_sides)
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n_sides, endpoint=False)
    for i in range(n_sides):
        points[i, 0] = center[0] + radius * np.cos(angles[i])
        points[i, 1] = center[1] + radius * np.sin(angles[i])
    
    # Add perturbation to escape local minima near perfect symmetry
    perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, (n_sides, 2))
    points += perturbation
    return points

# Helper function for custom initialization within a circle
def _custom_init_population_in_circle(popsize, n_points, n_dim, center, radius, seed):
    np.random.seed(seed) # Ensure custom init is reproducible
    initial_population = np.zeros((popsize, n_points * n_dim))

    # Number of structured initial configurations to include
    # Using a small number (e.g., 3) to introduce good starting points without dominating the population
    num_structured_inits = min(popsize, 3) 

    for i in range(popsize):
        if i < num_structured_inits and n_points == 16:
            # Generate a structured initial configuration (e.g., a perturbed 16-gon)
            # Place points slightly inside the radius (0.95*radius) to allow for perturbation
            # and then project them back if they go outside.
            points = _generate_regular_polygon_points(n_points, center, radius * 0.95, radius * 0.05, seed + i)
            
            # Ensure points are within the circle after perturbation
            for j in range(n_points):
                dist_sq = np.sum((points[j] - center)**2)
                if dist_sq > radius**2:
                    # Project point back to the boundary if it went outside
                    factor = radius / np.sqrt(dist_sq)
                    points[j] = center + (points[j] - center) * factor
        else:
            # Generate points uniformly within a circle by area (original method)
            points = np.zeros((n_points, n_dim))
            for j in range(n_points):
                r_val = radius * np.sqrt(np.random.rand())
                theta_val = 2 * np.pi * np.random.rand()
                points[j, 0] = center[0] + r_val * np.cos(theta_val)
                points[j, 1] = center[1] + r_val * np.sin(theta_val)
        initial_population[i] = points.flatten()
    return initial_population

def min_max_dist_dim2_16()->np.ndarray:
    """ 
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    This is achieved by using a global optimization algorithm (Differential Evolution)
    followed by a local refinement step (Nelder-Mead), with improved constraint handling.

    The key insight is that for a fixed area, a circular domain minimizes the maximum possible
    distance between points compared to a square domain. This helps maximize the d_min/d_max
    ratio. Therefore, we constrain the points to lie within a circle using a hard penalty function
    (returning np.inf) in the optimizer's objective.
    The initial population for Differential Evolution is also generated within this circular domain.

    Returns:
        points (np.ndarray): An array of shape (16, 2) containing the (x, y) coordinates.
    """
    n_points = 16
    n_dim = 2
    
    # Define the circular domain parameters
    circle_center = np.array([0.5, 0.5])
    circle_radius = 0.5
    circle_radius_squared = circle_radius**2

    # The objective function to be MINIMIZED.
    # We want to maximize d_min / d_max, so we minimize its negative.
    # A hard penalty (np.inf) is added to enforce a circular boundary.
    def objective_func(x: np.ndarray) -> float:
        # Reshape the flat input array into a (n_points, n_dim) array of coordinates
        points = x.reshape(n_points, n_dim)
        
        # --- Circular Constraint Enforcement (Death Penalty) ---
        # Calculate squared distance from center for all points
        radii_squared = np.sum((points - circle_center)**2, axis=1)
        
        # Calculate how much points are outside the circle.
        # Use a soft quadratic penalty instead of a hard death penalty (np.inf).
        # This makes the optimization landscape smoother and allows the optimizer
        # to recover from slightly invalid configurations.
        # Tuned penalty_factor: Reduced from 10000.0 to 1000.0 to make it less dominant
        # relative to the primary objective, allowing for better exploration of the boundary.
        penalty_factor = 1000.0 
        outside_radius_diff = np.maximum(0, radii_squared - circle_radius_squared)
        constraint_penalty = np.sum(outside_radius_diff**2) * penalty_factor
        
        # Calculate all pairwise distances efficiently
        distances = pdist(points)
        
        # Handle edge cases for distances.
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Penalize truly coincident points (d_min very close to zero, which would lead to an undefined or zero ratio).
        # This check is crucial to prevent numerical instability and ensure distinct points are sought.
        if d_min < 1e-12: # Use a very small threshold for floating point 'zero' distance to detect coincident points
            return 1e10 + constraint_penalty # Return a very high penalty if points are effectively coincident
        
        # If d_max is zero, it means all points are coincident. This should be caught
        # by the d_min check, but as a safeguard.
        if d_max == 0:
            return 1e10 + constraint_penalty

        # The objective to minimize: -ratio + constraint_penalty
        # This combines the primary objective with the constraint enforcement.
        return -d_min / d_max + constraint_penalty

    # Define box bounds [0,1]x[0,1] which fully contain our target circle.
    # This is still required by differential_evolution, but our custom init and
    # death penalty ensure points stay within the circle.
    bounds = [(0, 1)] * (n_points * n_dim)
    
    # Use a fixed seed for reproducibility.
    seed = 42

    # Generate initial population within the circular domain for Differential Evolution
    # Increase popsize for better global exploration. A common heuristic is 10*D where D is dimensions.
    # n_points * n_dim = 16 * 2 = 32 dimensions. So popsize ~320.
    # Balancing with eval_time, let's use 100 which is a significant increase from 30.
    de_popsize = 100
    de_maxiter = 6000 # Increased iterations for more thorough search (from 4000)
    de_mutation = (0.5, 1.0) # Standard mutation range, (0.5, 1.5) can be too aggressive

    # Store the best result across multiple restarts of Differential Evolution
    best_de_objective_value = np.inf
    best_de_x = None

    num_de_restarts = 3 # Reduced number of restarts (from 5) to balance eval_time with increased maxiter

    for i in range(num_de_restarts):
        current_seed = seed + i # Vary seed for each restart for different initializations
        
        # Generate initial population within the circular domain for this restart
        initial_population_current = _custom_init_population_in_circle(
            popsize=de_popsize,
            n_points=n_points,
            n_dim=n_dim,
            center=circle_center,
            radius=circle_radius,
            seed=current_seed
        )

        # Run Differential Evolution
        current_result = differential_evolution(
            func=objective_func,
            bounds=bounds,
            strategy='best1bin',
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-7,
            mutation=de_mutation,
            recombination=0.8,
            seed=current_seed,
            disp=False,
            init=initial_population_current
        )
        
        # Check if this run yielded a better objective value
        if current_result.fun < best_de_objective_value:
            best_de_objective_value = current_result.fun
            best_de_x = current_result.x

    # Ensure best_de_x is not None. In a multi-start scenario, it should be set in the loop.
    # If for some reason all runs failed to find a valid solution (objective_func always returned 1e10),
    # then best_de_x could be None. This is unlikely with the soft penalty.
    # As a safeguard, let's assume at least one run produces a finite result.
    
    # The best solution from Differential Evolution across all restarts
    optimal_points_de = best_de_x.reshape(n_points, n_dim)

    # Refine the solution using a local optimizer (Nelder-Mead)
    # Nelder-Mead is suitable for non-smooth functions and doesn't require gradients.
    # The objective_func now uses a soft penalty for constraints, which works with Nelder-Mead.
    refined_result = minimize(
        fun=objective_func,
        x0=optimal_points_de.flatten(), # Start from the best DE solution
        method='Nelder-Mead',
        options={'maxiter': 2000, 'disp': False} # Increased iterations for local refinement
    )
    optimal_points = refined_result.x.reshape(n_points, n_dim)

    return optimal_points
# EVOLVE-BLOCK-END