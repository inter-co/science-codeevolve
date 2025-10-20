# EVOLVE-BLOCK-START
import numpy as np 
from scipy.spatial.distance import pdist, squareform # Added squareform
import random # Added random module for acceptance criterion and specific move choices

# Helper function to calculate the min/max ratio (kept identical as it was already good)
def calculate_min_max_ratio(points: np.ndarray) -> float:
    """
    Calculates the ratio of minimum to maximum distance for a set of points.
    
    Args:
        points: A numpy array of shape (N, D) representing N points in D dimensions.

    Returns:
        The ratio of minimum to maximum pairwise Euclidean distance. Returns 0.0 if
        there are fewer than 2 points or if all points are identical.
    """
    if points.shape[0] < 2:
        return 0.0

    # Compute pairwise Euclidean distances using pdist for efficiency
    distances = pdist(points, 'euclidean')

    if len(distances) == 0: # Should not happen with 2+ points
        return 0.0

    dmin = np.min(distances)
    dmax = np.max(distances)

    if dmax == 0: # All points are identical (dmin will also be 0)
        return 0.0
    
    return dmin / dmax

def min_max_dist_dim3_14()->np.ndarray:
    """ 
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    This function implements an advanced Simulated Annealing (SA) algorithm with geometric 
    initialization and hybrid perturbation strategies to find an optimal arrangement of 
    14 points within a [0,1]^3 unit cube, maximizing the min/max distance ratio.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points
                that achieved the highest min/max ratio during the optimization.
    """

    n = 14
    d = 3

    # --- Tuned Simulated Annealing Optimization Parameters (from Inspiration 3) ---
    T_start = 0.05       # Lower starting temperature to preserve initial structure.
    T_end = 1e-8         # Very low final temperature for maximal precision.
    max_iterations = 1_500_000 # Further increased iterations for thorough fine-tuning.
    # Recalculate alpha for the new T_start and max_iterations to ensure slow cooling.
    alpha = (T_end / T_start)**(1 / max_iterations) # Approx 0.9999905
    initial_step_size = 0.01 # Much smaller initial steps for gentle adjustments.
    min_step_size = 1e-8   # Allow for extremely fine final tuning.
    
    # Recalculate distance matrix more frequently for more accurate targeted moves,
    # especially critical in the initial phases where points might move more.
    recalc_dist_interval = 25 

    # Probabilities for hybrid perturbation strategy
    dmin_prob = 0.15 # Probability of pushing closest points apart (Exploitation)
    dmax_prob = 0.15 # Probability of pulling farthest points together (Exploitation)
    
    optimization_seed = 42 # Random seed for reproducibility

    # Ensure reproducibility for both numpy and standard random module
    np.random.seed(optimization_seed)
    random.seed(optimization_seed)

    # --- Initialization Strategy: Cube + Octahedron Vertices (from Inspiration) ---
    # This provides a good symmetric starting point for 14 points, formed by a cuboctahedron-like
    # arrangement. It's a strong heuristic for 14 points, often outperforming random starts.
    cube_verts = np.array([[-1,-1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,1,1]])
    octa_verts = np.array([[-2,0,0], [2,0,0], [0,-2,0], [0,2,0], [0,0,-2], [0,0,2]])
    initial_points = np.vstack([cube_verts, octa_verts])
    
    # Scale and translate to fit within the unit cube [0,1]^3
    min_coord, max_coord = initial_points.min(), initial_points.max()
    current_points = (initial_points - min_coord) / (max_coord - min_coord)
    
    # Add a small amount of noise to break perfect symmetry and allow exploration
    current_points += np.random.uniform(-0.001, 0.001, size=current_points.shape)
    current_points = np.clip(current_points, 0, 1)

    # Evaluate the initial state
    current_ratio = calculate_min_max_ratio(current_points)
    best_points = np.copy(current_points)
    best_ratio = current_ratio

    T = T_start # Initialize temperature
    
    # Initialize distance matrix variables to avoid recalculating every iteration
    distances_condensed = None
    dist_matrix = None

    # --- Simulated Annealing Loop ---
    for i in range(max_iterations):
        # Adaptive step size: starts large, decreases with temperature
        step_size = max(initial_step_size * (T / T_start), min_step_size)
        new_points = np.copy(current_points)
        
        # --- Advanced Hybrid Perturbation Strategy (from Inspiration) ---
        # This strategy uses three types of moves to balance exploration and exploitation.
        # Optimized to compute the expensive distance matrix only when necessary.
        rand_choice = random.random()

        # Recalculate distance matrix periodically for targeted moves, or if not yet initialized
        if i % recalc_dist_interval == 0 or distances_condensed is None:
            distances_condensed = pdist(current_points)
            dist_matrix = squareform(distances_condensed)

        if rand_choice < dmin_prob + dmax_prob:
            # Use the most recent distance matrix for targeted moves
            if rand_choice < dmin_prob:
                # Greedy Move: Push apart the two closest points to directly improve d_min.
                # Ensure dist_matrix is available, if not, fall back to random
                if dist_matrix is not None:
                    # Create a temporary copy to avoid modifying the stored dist_matrix for subsequent iterations
                    temp_dist_matrix = np.copy(dist_matrix)
                    np.fill_diagonal(temp_dist_matrix, np.inf) # Temporarily set diagonal to inf
                    i_min, j_min = np.unravel_index(np.argmin(temp_dist_matrix), temp_dist_matrix.shape)
                    
                    point_idx = random.choice([i_min, j_min])
                    other_point = current_points[j_min if point_idx == i_min else i_min]
                    
                    repulsive_vec = new_points[point_idx] - other_point
                    norm = np.linalg.norm(repulsive_vec)
                    delta = repulsive_vec / norm * step_size if norm > 1e-9 else np.zeros(d)
                else: # Fallback if dist_matrix somehow wasn't initialized
                    point_idx = np.random.randint(n)
                    delta = np.random.uniform(-step_size, step_size, size=d)
            else:
                # Compressing Move: Pull together the two farthest points to directly decrease d_max.
                # Ensure dist_matrix is available, if not, fall back to random
                if dist_matrix is not None:
                    i_max, j_max = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                    
                    point_idx = random.choice([i_max, j_max])
                    other_point = current_points[j_max if point_idx == i_max else i_max]
                    
                    # Create a vector pointing from the chosen point TOWARDS the other one.
                    attractive_vec = other_point - new_points[point_idx]
                    norm = np.linalg.norm(attractive_vec)
                    delta = attractive_vec / norm * step_size if norm > 1e-9 else np.zeros(d)
                else: # Fallback if dist_matrix somehow wasn't initialized
                    point_idx = np.random.randint(n)
                    delta = np.random.uniform(-step_size, step_size, size=d)
        else:
            # Random Move: Standard SA exploration, computationally cheap.
            point_idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size, size=d)

        # Apply perturbation and enforce unit cube constraints
        new_points[point_idx] += delta
        new_points[point_idx] = np.clip(new_points[point_idx], 0, 1)

        new_ratio = calculate_min_max_ratio(new_points)

        # Metropolis-Hastings acceptance criterion for maximization (using random.random())
        if new_ratio > current_ratio or random.random() < np.exp((new_ratio - current_ratio) / T):
            current_points = new_points
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = np.copy(current_points)
        
        # Cooling schedule: Decrease temperature
        T = max(T * alpha, T_end) # Ensure T does not go below T_end
        
    return best_points
# EVOLVE-BLOCK-END