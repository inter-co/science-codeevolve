# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import math
from deap import base, creator, tools, algorithms
from scipy.optimize import minimize
import numba # Added numba for performance

# --- Configuration Constants ---
N_CIRCLES = 26
UNIT_SQUARE_SIZE = 1.0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# GA Parameters
POP_SIZE = 500   # Increased population size to leverage available time and improve exploration
NGEN = 1000      # Increased generations for better convergence and exploitation
CXPB = 0.7       # Crossover probability
MUTPB = 0.3      # Mutation probability
IND_MUT_PB = 0.05 # Independent probability for each attribute to be mutated

# Penalty coefficients for constraint violations
# Increased overlap penalty to more strongly discourage infeasible solutions.
PENALTY_OVERLAP_FACTOR = 200000.0
PENALTY_BOUNDARY_FACTOR = 1000.0

# Bounds for circle parameters (x, y, r)
MIN_X, MAX_X = 0.0, UNIT_SQUARE_SIZE
MIN_Y, MAX_Y = 0.0, UNIT_SQUARE_SIZE
MIN_R, MAX_R = 1e-4, 0.5 # Radius must be positive; max possible is 0.5 for a single circle

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize sum of radii
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator for x, y, r
# Use a perturbed grid initialization for a better starting configuration.
# This spreads circles out, reducing initial overlaps and guiding the search.
def generate_initial_param():
    params = []
    grid_size = 6
    spacing = UNIT_SQUARE_SIZE / grid_size
    
    # Generate all possible grid cell indices
    all_indices = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    
    # Randomly choose 26 unique cell indices
    chosen_indices = random.sample(all_indices, N_CIRCLES)
    
    for i, j in chosen_indices:
        # Center of the cell + random perturbation
        x = (i + 0.5) * spacing + random.uniform(-spacing/4, spacing/4)
        y = (j + 0.5) * spacing + random.uniform(-spacing/4, spacing/4)
        r = random.uniform(0.01, 0.05) # Small initial radius
        params.extend([x, y, r])
        
    return params

toolbox.register("individual", tools.initIterate, creator.Individual, generate_initial_param)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Helper Functions for Fitness Evaluation ---

@numba.jit(nopython=True) # JIT compile for performance
def _calculate_distances_sq(circles):
    """Calculates squared distances between centers of all circle pairs."""
    centers = circles[:, :2]
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    return dx**2 + dy**2

@numba.jit(nopython=True) # JIT compile for performance
def _evaluate_packing(individual):
    """
    Fitness function for the GA. Maximizes sum of radii while penalizing overlaps and boundary violations.
    """
    # Reshape individual (list) into a numpy array of circles directly within the jitted function
    circles = np.array(individual).reshape(N_CIRCLES, 3)
    radii = circles[:, 2]
    
    # Ensure radii are positive. Numba compatible check.
    if np.any(radii <= 0.0): # Use 0.0 for float comparison in Numba
        return -np.inf, # Use np.inf for Numba compatibility

    sum_radii = np.sum(radii)
    
    # --- Overlap Penalty ---
    distances_sq = _calculate_distances_sq(circles)
    
    # Calculate overlap violations using squared distances (as suggested in problem description)
    # violation_sq = max(0, (ri + rj)**2 - ((xi-xj)**2 + (yi-yj)**2))
    required_min_dist_sq = (radii[:, np.newaxis] + radii[np.newaxis, :])**2
    overlap_violations_sq = np.maximum(0.0, required_min_dist_sq - distances_sq) # Use 0.0 for float comparison
    
    # Only consider upper triangle to avoid double counting and self-overlap.
    # Manual loop for Numba compatibility as np.triu is not fully supported for sum in nopython mode.
    total_overlap_violation = 0.0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            total_overlap_violation += overlap_violations_sq[i, j]
    
    # --- Boundary Penalty ---
    centers = circles[:, :2]
    
    # Check x-axis boundaries
    x_min_violation = np.maximum(0.0, radii - centers[:, 0])
    x_max_violation = np.maximum(0.0, centers[:, 0] + radii - UNIT_SQUARE_SIZE)
    
    # Check y-axis boundaries
    y_min_violation = np.maximum(0.0, radii - centers[:, 1])
    y_max_violation = np.maximum(0.0, centers[:, 1] + radii - UNIT_SQUARE_SIZE)
    
    total_boundary_violation = np.sum(x_min_violation + x_max_violation + y_min_violation + y_max_violation)
    
    # Calculate total penalty
    total_penalty = PENALTY_OVERLAP_FACTOR * total_overlap_violation + \
                    PENALTY_BOUNDARY_FACTOR * total_boundary_violation
    
    # Fitness is sum_radii minus penalties. We want to maximize this.
    fitness_value = sum_radii - total_penalty
    
    return fitness_value,

toolbox.register("evaluate", _evaluate_packing)

# GA Operators
# Increased eta for more exploitation of promising regions, as we have fewer generations.
# Increased tournament size for higher selection pressure to speed up convergence.
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[MIN_X, MIN_Y, MIN_R] * N_CIRCLES,
                 up=[MAX_X, MAX_Y, MAX_R] * N_CIRCLES, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[MIN_X, MIN_Y, MIN_R] * N_CIRCLES,
                 up=[MAX_X, MAX_Y, MAX_R] * N_CIRCLES, eta=20.0, indpb=IND_MUT_PB)
toolbox.register("select", tools.selTournament, tournsize=7)

# --- Local Optimization (Post-GA Refinement) ---

def _objective_func_scipy(params):
    """Objective function for scipy.optimize: negative sum of radii (to maximize sum_radii)."""
    radii = params[2::3]
    return -np.sum(radii)

def _constraint_overlap(params, i, j):
    """Non-overlap constraint: distance between centers must be >= sum of radii."""
    x_i, y_i, r_i = params[i*3:i*3+3]
    x_j, y_j, r_j = params[j*3:j*3+3]
    
    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
    min_dist_sq = (r_i + r_j)**2
    
    return dist_sq - min_dist_sq # Must be >= 0

def _constraint_boundary_x_min(params, k):
    """Boundary constraint: x - r >= 0."""
    x, _, r = params[k*3:k*3+3]
    return x - r

def _constraint_boundary_x_max(params, k):
    """Boundary constraint: 1 - x - r >= 0."""
    x, _, r = params[k*3:k*3+3]
    return UNIT_SQUARE_SIZE - x - r

def _constraint_boundary_y_min(params, k):
    """Boundary constraint: y - r >= 0."""
    _, y, r = params[k*3:k*3+3]
    return y - r

def _constraint_boundary_y_max(params, k):
    """Boundary constraint: 1 - y - r >= 0."""
    _, y, r = params[k*3:k*3+3]
    return UNIT_SQUARE_SIZE - y - r

def refine_packing_with_scipy(initial_circles_flat, max_iter=2000):
    """
    Refines a packing using scipy.optimize.minimize with SLSQP.
    """
    bounds = [(MIN_X, MAX_X), (MIN_Y, MAX_Y), (MIN_R, MAX_R)] * N_CIRCLES
    
    constraints = []
    # Add non-overlap constraints
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            constraints.append({'type': 'ineq', 'fun': _constraint_overlap, 'args': (i, j)})
    
    # Add boundary constraints
    for k in range(N_CIRCLES):
        constraints.append({'type': 'ineq', 'fun': _constraint_boundary_x_min, 'args': (k,)})
        constraints.append({'type': 'ineq', 'fun': _constraint_boundary_x_max, 'args': (k,)})
        constraints.append({'type': 'ineq', 'fun': _constraint_boundary_y_min, 'args': (k,)})
        constraints.append({'type': 'ineq', 'fun': _constraint_boundary_y_max, 'args': (k,)})
        
    # Ensure all radii are positive
    for k in range(N_CIRCLES):
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=k: p[idx*3+2] - MIN_R}) # r >= MIN_R

    # Initial guess should be valid or close to valid
    # The GA solution `best_ind` is our starting point
    # Ensure initial radii are at least MIN_R to satisfy the constraint
    initial_guess = np.array(initial_circles_flat)
    initial_guess[2::3] = np.maximum(initial_guess[2::3], MIN_R)

    # Perform the optimization
    # Using 'SLSQP' as it handles bounds and general non-linear inequality constraints
    # Increased maxiter for more thorough local refinement, leveraging available time.
    res = minimize(_objective_func_scipy, initial_guess, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 100, 'ftol': 1e-8, 'disp': False})

    if not res.success:
        # If optimization failed, return the original best GA solution
        # Or try to repair it slightly, but for now, just return it.
        # print(f"Scipy optimization failed: {res.message}")
        return np.array(initial_circles_flat).reshape(N_CIRCLES, 3) # Directly reshape
    
    return res.x.reshape(N_CIRCLES, 3) # Directly reshape


def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a Genetic Algorithm followed by local optimization (SLSQP).

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()

    # Create an initial population
    pop = toolbox.population(n=POP_SIZE)
    
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Initialize statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame to store the best individual
    hof = tools.HallOfFame(1)

    # Run the Genetic Algorithm
    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=False)

    best_individual_ga = hof[0]
    best_circles_ga = np.array(best_individual_ga).reshape(N_CIRCLES, 3) # Directly reshape
    
    # print(f"GA finished. Best sum_radii (penalized): {best_individual_ga.fitness.values[0]}")
    # print(f"Raw sum_radii from GA best: {np.sum(best_circles_ga[:, 2])}")

    # --- Post-processing: Local Optimization ---
    # Only refine if the GA found a reasonably good (low-penalty) solution
    # The fitness value includes penalties, so a high fitness means good sum_radii and low penalties.
    # If the best_individual_ga.fitness.values[0] is still negative (due to high penalties),
    # it means the GA couldn't find a feasible solution, so local refinement on such a solution might be problematic.
    # However, SLSQP can handle initial infeasible points, so we can try.
    
    refined_circles = refine_packing_with_scipy(best_individual_ga)
    
    # Final check for radii validity after refinement
    refined_circles[:, 2] = np.maximum(refined_circles[:, 2], MIN_R)

    # Re-evaluate the refined solution to get its true sum_radii without penalties
    # This is for reporting, not for GA fitness.
    final_sum_radii = np.sum(refined_circles[:, 2])

    # print(f"Refined sum_radii: {final_sum_radii}")

    return refined_circles

# EVOLVE-BLOCK-END