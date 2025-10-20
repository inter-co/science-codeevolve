# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import pdist

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by framing it as a constrained optimization problem and using SLSQP.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    np.random.seed(42)  # For reproducibility

    # Helper functions for optimization, defined locally to avoid polluting the global namespace.
    def objective(params, n):
        """The objective is to maximize the sum of radii, so we minimize its negative."""
        radii = params[2*n:]
        return -np.sum(radii)

    def constraints_func(params, n):
        """
        Defines the constraints for the optimization problem.
        All returned values must be non-negative (>= 0) for the constraint to be satisfied.
        """
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        
        # 1. Boundary constraints:
        # r_i <= x_i <= 1 - r_i  => x_i - r_i >= 0 and 1 - x_i - r_i >= 0
        # r_i <= y_i <= 1 - r_i  => y_i - r_i >= 0 and 1 - y_i - r_i >= 0
        boundary_cons = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])
        
        # 2. Non-overlap constraints:
        # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        # This is computationally better as (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        coords = np.vstack([x, y]).T
        dist_sq = pdist(coords, 'sqeuclidean') # Efficiently calculate squared Euclidean distances for all pairs
        
        # Vectorized calculation of (ri + rj)^2 for all pairs
        indices = np.triu_indices(n, k=1)
        r_i = r[indices[0]]
        r_j = r[indices[1]]
        sum_r_sq = (r_i + r_j)**2
        
        overlap_cons = dist_sq - sum_r_sq

        return np.concatenate([boundary_cons, overlap_cons])

    # 1. Generate a good, feasible initial guess
    # Start with a slightly perturbed 5x6 grid, taking the first 26 points.
    rows, cols = 5, 6
    x_grid, y_grid = np.mgrid[0:rows, 0:cols]
    
    x_coords = (x_grid.flatten()[:n].astype(float) + 0.5) / cols
    y_coords = (y_grid.flatten()[:n].astype(float) + 0.5) / rows
    
    # Perturb positions slightly to break symmetry and aid optimization
    x_coords += np.random.uniform(-0.01, 0.01, n)
    y_coords += np.random.uniform(-0.01, 0.01, n)
    
    # Initialize with small radii to ensure initial feasibility
    radii = np.full(n, 0.05)
    
    x0 = np.concatenate([x_coords, y_coords, radii])

    # 2. Define bounds for each variable (x, y, r)
    bounds_x = [(0, 1)] * n
    bounds_y = [(0, 1)] * n
    # Set a minimum radius to prevent circles from vanishing
    bounds_r = [(0.01, 0.5)] * n
    bounds = bounds_x + bounds_y + bounds_r

    # 3. Define the constraints dictionary for the optimizer
    constraints = [{'type': 'ineq', 'fun': constraints_func, 'args': (n,)}]

    # 4. Run the optimization. Using basinhopping for global search.
    # Define minimizer_kwargs for the local optimizer (SLSQP)
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": constraints,
        "args": (n,),
        "options": {'maxiter': 1000, 'ftol': 1e-9, 'disp': False} # Increased maxiter and tightened ftol for higher precision local search
    }

    # Run basinhopping
    # niter: number of basin-hopping steps. Each step performs a local optimization.
    # T: temperature for the metropolis criterion. Higher T means more exploration.
    # stepsize: maximum step size for random perturbations.
    # seed: For reproducibility.
    result = basinhopping(
        objective, 
        x0, 
        minimizer_kwargs=minimizer_kwargs, 
        niter=250, # Further increased iterations for more thorough global search
        T=2.0,     # Increased temperature for broader exploration
        stepsize=0.05, # Default step size, reasonable perturbation
        seed=42    # For reproducibility
    )

    # 5. Reshape the flat result vector back into the (n, 3) circle format
    final_params = result.x
    circles = np.column_stack([
        final_params[:n],       # x coordinates
        final_params[n:2*n],    # y coordinates
        final_params[2*n:]      # radii
    ])

    return circles

# EVOLVE-BLOCK-END