# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

# Helper function to calculate min_max_ratio
def _calculate_min_max_ratio(points: np.ndarray) -> float:
    """
    Calculates the ratio of minimum to maximum pairwise distance for a set of 3D points.
    """
    if points.shape[0] < 2:
        return 0.0
    
    # Ensure points are 3D
    if points.shape[1] != 3:
        raise ValueError("Points must be 3D.")

    # Calculate pairwise distances using pdist for efficiency
    distances = pdist(points)

    if len(distances) == 0: # Should not happen for N >= 2
        return 0.0

    dmin = np.min(distances)
    dmax = np.max(distances)

    # Avoid division by zero if all points are identical (dmax = 0)
    if dmax == 0:
        return 0.0
    
    return dmin / dmax

# Objective function for optimization
# It will be minimized, so we return the negative ratio
def _objective_function(coords_flat: np.ndarray) -> float:
    """
    Objective function for the optimizer. Takes flattened coordinates, reshapes them,
    and returns the negative of the min_max_ratio to enable minimization.
    """
    n = len(coords_flat) // 3
    points = coords_flat.reshape(n, 3)
    
    # The spherical constraint is handled by `minimize` itself, so no projection here.
    ratio = _calculate_min_max_ratio(points)
    return -ratio

from scipy.optimize import basinhopping # New import for global optimization

# Constraint function: each point must be within or on the unit sphere
def _sphere_inequality_constraints(flat_points: np.ndarray) -> np.ndarray:
    """
    Returns an array of ||p_i||^2 - 1 for each point. Should be <= 0 for points
    to be within or on the unit sphere.
    """
    points = flat_points.reshape((-1, 3))
    return np.sum(points**2, axis=1) - 1.0

# Constraint Jacobian for the inequality constraint (can improve SLSQP performance)
def _sphere_inequality_constraints_jac(flat_points: np.ndarray) -> np.ndarray:
    """
    Jacobian of the sphere inequality constraints.
    For c_i = x_i^2 + y_i^2 + z_i^2 - 1, the gradient w.r.t. (x_i, y_i, z_i) is (2x_i, 2y_i, 2z_i).
    """
    points = flat_points.reshape((-1, 3))
    n_points = points.shape[0]
    jac = np.zeros((n_points, n_points * 3))
    for i in range(n_points):
        jac[i, i*3:(i+1)*3] = 2 * points[i]
    return jac

# Initial points generator using a bicapped hexagonal antiprism (from Inspiration 1 & 2)
def _get_best_initial_guess(n: int, seed: int) -> np.ndarray:
    """
    Generates an initial configuration of 14 points based on a bicapped hexagonal antiprism.
    All points are initially placed on the unit sphere.
    """
    if n != 14:
        raise ValueError("This initial guess is specifically for 14 points.")

    points = []

    # Two poles (caps) at z=1 and z=-1
    points.append([0, 0, 1])
    points.append([0, 0, -1])

    # Two hexagonal layers
    # A value of h=0.5 places the hexagonal layers at z=0.5 and z=-0.5.
    # This results in a reasonably symmetric initial configuration with a good starting ratio.
    h = 0.5 
    r_h = np.sqrt(max(0, 1.0 - h**2)) # Radius of the hexagons at height h to keep points on unit sphere

    # Top hexagon (6 points)
    for i in range(6):
        angle = i * 2 * np.pi / 6 # 60 degree increments
        x = r_h * np.cos(angle)
        y = r_h * np.sin(angle)
        points.append([x, y, h])

    # Bottom hexagon (6 points), rotated by pi/6 (30 degrees) relative to top
    for i in range(6):
        angle = i * 2 * np.pi / 6 + np.pi / 6 # Offset by 30 degrees
        x = r_h * np.cos(angle)
        y = r_h * np.sin(angle)
        points.append([x, y, -h])

    initial_points = np.array(points, dtype=float)
    
    # Ensure all points are exactly on the unit sphere after construction
    # This step is mainly for numerical robustness, as the construction should already place them on the sphere.
    norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
    # Handle cases where norm might be zero (extremely unlikely with this construction)
    initial_points = np.where(norms == 0, np.ones_like(initial_points), initial_points / norms)

    return initial_points

def min_max_dist_dim3_14()->np.ndarray:
    """ 
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    This function uses `basinhopping` for global optimization, initialized with a high-quality
    geometric guess (bicapped hexagonal antiprism). `SLSQP` is used as the local minimizer,
    with inequality constraints to keep points within or on the unit sphere.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3 # Dimensions

    # Set random seed for reproducibility for all stochastic components
    np.random.seed(42)

    # Generate initial points based on a symmetric geometric configuration (bicapped hexagonal antiprism)
    initial_points = _get_best_initial_guess(n, seed=42)
    x0 = initial_points.flatten()

    # Define local minimization method for basinhopping
    # SLSQP is suitable for constrained optimization, particularly for non-linear inequality constraints.
    minimizer_kwargs = {
        "method": "SLSQP",
        "constraints": [
            {'type': 'ineq', 'fun': _sphere_inequality_constraints, 'jac': _sphere_inequality_constraints_jac}
        ],
        "options": {"maxiter": 2000, "ftol": 1e-7} # SLSQP uses ftol for termination
    }

    # Use Basin Hopping for global optimization to escape local minima (from Inspiration 1 & 2)
    # This combines random perturbations (hopping) with local minimization.
    result = basinhopping(
        func=_objective_function,
        x0=x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=500,     # Number of global hopping iterations (increased for better exploration)
        T=1.5,         # Temperature parameter: higher T allows larger, less likely jumps
        stepsize=0.75, # Maximum step size for random perturbations
        seed=42,       # Seed for basinhopping's internal random number generator (reproducibility)
    )

    optimized_flat_points = result.x
    optimized_points = optimized_flat_points.reshape((n, d))

    # Post-optimization processing (from Inspiration 1 & 2):
    # 1. Center the points around the origin (should be minimal if initial points were centered
    #    and constraints maintain symmetry, but good for robustness).
    optimized_points -= np.mean(optimized_points, axis=0)
    
    # 2. Scale the points to fit within a unit sphere (max distance from origin = 1).
    #    This ensures consistency with the problem's typical normalization requirements,
    #    where dmax is typically the diameter of the unit sphere (2.0).
    #    The inequality constraints ensure points are *within* or *on* the unit sphere, but
    #    a final scaling ensures the outermost point is exactly at radius 1.
    max_radius = np.max(np.linalg.norm(optimized_points, axis=1))
    if max_radius > 1e-6: # Avoid division by zero if all points are at origin
        optimized_points /= max_radius

    return optimized_points
# EVOLVE-BLOCK-END