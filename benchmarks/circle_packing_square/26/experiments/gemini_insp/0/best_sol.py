# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping # Added basinhopping
from scipy.spatial.distance import pdist

def circle_packing26() -> np.ndarray:
    """
    Finds an optimal arrangement of 26 non-overlapping circles in a unit square
    to maximize the sum of their radii using a hybrid basinhopping + SLSQP approach.

    The problem is formulated as a constrained optimization problem:
    - Objective: Maximize sum(radii)
    - Variables: Circle centers (x_i, y_i) and radii (r_i)
    - Constraints:
        1. Each circle is fully contained within the [0,1]x[0,1] square.
        2. No two circles overlap.

    Returns:
        np.ndarray: An array of shape (26, 3) where each row represents a
                    circle as [x_center, y_center, radius].
    """
    n = 26

    def objective(params: np.ndarray) -> float:
        """The objective function to minimize: the negative sum of radii."""
        return -np.sum(params[2::3])

    def constraints(params: np.ndarray) -> np.ndarray:
        """
        Returns a vector of constraint violations.
        For SLSQP, constraints are of the form C(x) >= 0.
        """
        p = params.reshape((n, 3))
        centers, radii = p[:, :2], p[:, 2]
        x, y = centers[:, 0], centers[:, 1]

        # 1. Containment constraints (4*n constraints)
        # c_i >= 0 => r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        containment_c = np.concatenate([
            x - radii,
            1 - x - radii,
            y - radii,
            1 - y - radii
        ])

        # 2. Non-overlap constraints (n*(n-1)/2 constraints)
        # dist(ci, cj)^2 >= (ri + rj)^2
        if n > 1:
            # Vectorized calculation for performance
            dist_sq = pdist(centers)**2
            # Create condensed matrix of (r_i + r_j)
            radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            radii_sum_condensed = radii_sum[np.triu_indices(n, k=1)]
            overlap_c = dist_sq - radii_sum_condensed**2
        else:
            overlap_c = np.array([])
            
        return np.concatenate([containment_c, overlap_c])

    # Initial guess (x0): a feasible grid-like arrangement
    # A 6x5 grid provides 30 spots; we use the first 26.
    nx, ny = 6, 5
    x_coords = np.linspace(1 / (2 * nx), 1 - 1 / (2 * nx), nx)
    y_coords = np.linspace(1 / (2 * ny), 1 - 1 / (2 * ny), ny)
    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    
    x0 = np.zeros(n * 3)
    initial_centers = grid_points[:n]
    # Small initial radius to ensure the initial guess is feasible
    initial_radius = 0.08
    x0[0::3] = initial_centers[:, 0]
    x0[1::3] = initial_centers[:, 1]
    x0[2::3] = initial_radius

    # Bounds for variables: 0 <= x,y <= 1 and a small positive lower bound for r
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-6, 0.5)])

    # All constraints for SLSQP are inequality constraints (C(x) >= 0)
    cons_slsqp = [{'type': 'ineq', 'fun': constraints}]

    # Configure the local minimizer that basinhopping will call
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": cons_slsqp,
        "options": {'maxiter': 100, 'ftol': 1e-7, 'disp': False} # Reduced maxiter for inner SLSQP runs
    }

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configure the local minimizer that basinhopping will call
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": cons_slsqp,
        # Increased maxiter for inner SLSQP runs to allow for more thorough local convergence
        "options": {'maxiter': 200, 'ftol': 1e-7, 'disp': False}
    }

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run basinhopping for global exploration
    # Increased niter to allow for more extensive global search and more local optimizations
    # T: Temperature parameter (higher T allows for larger jumps/acceptance of worse solutions)
    # stepsize: The maximum step size for random perturbations
    res_bh = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs,
                          niter=200, T=1.0, stepsize=0.1, seed=42, disp=False)

    if not res_bh.lowest_optimization_result.success:
        print(f"Warning (Basinhopping local minimizer): The best local optimization found by basinhopping may not have fully converged. Message: {res_bh.lowest_optimization_result.message}")

    # The optimal parameters are stored in res_bh.x
    return res_bh.x.reshape((n, 3))

# EVOLVE-BLOCK-END