# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds

# Set a fixed random seed for reproducibility to ensure consistent results
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26

    # 1. Define the objective function (negative sum of radii for minimization)
    def objective(params: np.ndarray) -> float:
        """Calculates the negative sum of radii from the flattened parameter array."""
        circles_data = params.reshape(n, 3)
        radii = circles_data[:, 2]
        return -np.sum(radii)

    # 2. Define constraint functions
    # All constraints are of type 'ineq', meaning fun(params) >= 0.

    def containment_constraints(params: np.ndarray) -> np.ndarray:
        """
        Ensures all circles are fully contained within the unit square [0,1]x[0,1].
        Constraints: r <= x, x <= 1-r, r <= y, y <= 1-r.
        Transformed to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0.
        """
        circles_data = params.reshape(n, 3)
        x, y, r = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]
        
        return np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

    def non_overlap_constraints(params: np.ndarray) -> np.ndarray:
        """
        Ensures no two circles overlap.
        Constraint: distance((xi,yi), (xj,yj)) >= ri + rj.
        Transformed to: sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0.
        Vectorized computation for efficiency.
        """
        circles_data = params.reshape(n, 3)
        x, y, r = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]
        
        # Calculate all pairwise differences in x and y coordinates using broadcasting
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        
        # Calculate all pairwise squared Euclidean distances
        dist_sq = dx**2 + dy**2
        
        # Calculate all pairwise sums of radii
        r_sum = r[:, np.newaxis] + r
        
        # Extract the upper triangular part (excluding the diagonal) to get unique pairs
        # and avoid self-comparison (distance = 0, r_sum = 2r)
        upper_tri_indices = np.triu_indices(n, k=1)
        
        # Compute the non-overlap constraints
        # sqrt(dist_sq) - r_sum >= 0
        constraints = np.sqrt(dist_sq[upper_tri_indices]) - r_sum[upper_tri_indices]
        
        return constraints

    # 3. Define bounds for each variable (x, y, r)
    # x_i, y_i must be in [0, 1]
    # r_i must be in [epsilon_r_min, 0.5] (0.5 is the maximum possible radius for a single circle)
    epsilon_r_min = 1e-6 # Minimum radius to avoid numerical issues and ensure positive radii
    
    bounds_list = []
    for _ in range(n):
        bounds_list.append((0.0, 1.0))          # x_i coordinate
        bounds_list.append((0.0, 1.0))          # y_i coordinate
        bounds_list.append((epsilon_r_min, 0.5)) # r_i radius
    bounds = Bounds(
        np.array([b[0] for b in bounds_list]),
        np.array([b[1] for b in bounds_list])
    )

    # 4. Define constraints for scipy.optimize.minimize
    constraints = [
        {'type': 'ineq', 'fun': containment_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints}
    ]

    # 5. Multi-start optimization
    # Add basinhopping to the imports
    from scipy.optimize import basinhopping

    # Optimization settings for the local minimizer (SLSQP)
    # maxiter and ftol are adjusted to allow for more thorough optimization within each local search.
    options = {'maxiter': 5000, 'ftol': 1e-8, 'disp': False} 

    # Initial guess for basinhopping: A single random configuration is sufficient,
    # as basinhopping will generate perturbations.
    initial_x = np.random.uniform(0.0, 1.0, n)
    initial_y = np.random.uniform(0.0, 1.0, n)
    # Radii are uniformly distributed within a small range to ensure initial validity
    # and allow circles to grow during optimization.
    initial_r = np.random.uniform(epsilon_r_min, 0.08, n) 

    # Ensure initial x,y are consistent with r bounds to start from a feasible region
    initial_x = np.clip(initial_x, initial_r, 1 - initial_r)
    initial_y = np.clip(initial_y, initial_r, 1 - initial_r)
    
    initial_guess = np.stack([initial_x, initial_y, initial_r], axis=1).flatten()

    # Define minimizer_kwargs for basinhopping, which will pass these arguments to the local optimizer (SLSQP)
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": constraints,
        "options": options
    }

    # Perform basinhopping global optimization
    # niter: Number of hopping iterations (local optimization calls after perturbation)
    # T: Temperature parameter for the Metropolis criterion (higher T allows accepting worse solutions)
    # stepsize: Maximum step size for random perturbations
    # seed: For reproducibility of the random perturbations
    bh_result = basinhopping(
        objective,
        initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=100, # Number of hopping iterations
        T=1.0,     # Temperature parameter
        stepsize=0.05, # Step size for random perturbations
        seed=42    # For reproducibility
    )

    # The result of basinhopping contains the best found solution in bh_result.x
    optimized_circles = bh_result.x.reshape(n, 3)
    
    # Post-processing: Ensure all radii are at least epsilon_r_min in the final output.
    # This guards against floating point inaccuracies that might push radii slightly below the bound.
    optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], epsilon_r_min)

    # basinhopping always returns a result, so no explicit check for -np.inf is needed here.
    return optimized_circles

# EVOLVE-BLOCK-END