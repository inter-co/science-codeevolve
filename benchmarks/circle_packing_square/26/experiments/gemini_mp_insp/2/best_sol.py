# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping

def circle_packing26()->np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This is achieved by framing the problem as a constrained optimization and using scipy's SLSQP solver
    within a basinhopping global optimization framework.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    seed = 42
    rng = np.random.default_rng(seed)

    # The parameters for optimization are a flat array: [x_coords, y_coords, r_coords]
    # Total variables: 3 * n = 78

    # 1. Objective function: Maximize sum of radii, which is equivalent to
    # minimizing the negative sum of radii.
    def objective(params):
        # Radii are the last 'n' parameters
        radii = params[2*n:]
        return -np.sum(radii)

    # Jacobian for the objective function (analytic gradient)
    def objective_jac(params):
        # Gradient is 0 for x, y coordinates and -1 for radii
        return np.concatenate([np.zeros(2*n), -np.ones(n)])

    # 2. Constraints (Vectorized for performance)
    def all_constraints_func(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]

        # Containment constraints (vectorized)
        containment_constraints = np.concatenate([
            x - r,      # x_i - r_i >= 0
            1 - x - r,  # 1 - x_i - r_i >= 0
            y - r,      # y_i - r_i >= 0
            1 - y - r   # 1 - y_i - r_i >= 0
        ])

        # Non-overlap constraints (vectorized)
        i, j = np.triu_indices(n, k=1)
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        sum_r = r[i] + r[j]
        non_overlap_constraints = dx**2 + dy**2 - sum_r**2
        
        return np.concatenate([containment_constraints, non_overlap_constraints])

    def all_constraints_jac_func(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]

        num_constraints = 4 * n + n * (n - 1) // 2
        num_variables = 3 * n
        jac = np.zeros((num_constraints, num_variables))

        # --- Vectorized Containment Jacobian ---
        idx = np.arange(n)
        # Derivatives w.r.t. x_k
        jac[idx, idx] = 1.0
        jac[idx + n, idx] = -1.0
        # Derivatives w.r.t. y_k
        jac[idx + 2*n, n + idx] = 1.0
        jac[idx + 3*n, n + idx] = -1.0
        # Derivatives w.r.t. r_k
        jac[idx, 2*n + idx] = -1.0
        jac[idx + n, 2*n + idx] = -1.0
        jac[idx + 2*n, 2*n + idx] = -1.0
        jac[idx + 3*n, 2*n + idx] = -1.0

        # --- Vectorized Non-overlap Jacobian ---
        i, j = np.triu_indices(n, k=1)
        
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        sum_r = r[i] + r[j]
        
        # Rows for non-overlap constraints start after the 4*n containment constraints
        row_idx = np.arange(4 * n, num_constraints)
        
        # Derivatives w.r.t x
        jac[row_idx, i] = 2 * dx
        jac[row_idx, j] = -2 * dx
        
        # Derivatives w.r.t y
        jac[row_idx, n + i] = 2 * dy
        jac[row_idx, n + j] = -2 * dy
        
        # Derivatives w.r.t r
        jac[row_idx, 2*n + i] = -2 * sum_r
        jac[row_idx, 2*n + j] = -2 * sum_r
                
        return jac
    
    # Pack the constraint functions into a single dictionary for minimize, providing analytic Jacobians
    cons = ({'type': 'ineq', 'fun': all_constraints_func, 'jac': all_constraints_jac_func})

    # 3. Bounds for each variable
    # 0 <= x_i, y_i <= 1
    # 0 <= r_i <= 0.5 (a single circle cannot have radius > 0.5)
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    # 4. Initial guess (x0)
    # A good initial guess is crucial. We start with a 5x5 grid for the first 25
    # circles, which is a near-optimal packing for N=25. The 26th is added randomly.
    grid_size = 5
    grid_step = 1.0 / grid_size
    
    x_centers = np.linspace(grid_step/2, 1 - grid_step/2, grid_size)
    y_centers = np.linspace(grid_step/2, 1 - grid_step/2, grid_size)
    xx, yy = np.meshgrid(x_centers, y_centers)

    x0_x = np.zeros(n)
    x0_y = np.zeros(n)
    x0_r = np.full(n, grid_step / 2) # Initial radius for all

    x0_x[:25] = xx.flatten()
    x0_y[:25] = yy.flatten()

    # Place the 26th circle randomly, but with a substantial initial radius
    # (same as other circles) to give it a better chance to grow during optimization.
    x0_x[25] = rng.random()
    x0_y[25] = rng.random()
    x0_r[25] = grid_step / 2 # Initialize with the same radius as the grid circles

    x0 = np.concatenate([x0_x, x0_y, x0_r])

    # 5. Run the optimization
    # To improve the chances of finding a global optimum, use basinhopping,
    # which combines a global stepping algorithm with local optimization (SLSQP in this case).
    
    # Parameters for the local SLSQP optimizer
    minimizer_kwargs = {
        "method": "SLSQP",
        "jac": objective_jac,
        "bounds": bounds,
        "constraints": cons,
        "options": {'maxiter': 1500, 'disp': False} # Increased maxiter for thorough local search
    }

    # Basinhopping parameters for global exploration
    # niter: Number of global hopping steps. Each step includes a local minimization.
    # T: Temperature parameter for the Metropolis criterion (controls acceptance of higher energy states).
    # stepsize: The maximum step size for random perturbations of variables.
    # seed: For reproducibility of the stochastic process.
    result = basinhopping(objective, x0,
                          minimizer_kwargs=minimizer_kwargs,
                          niter=100, # Number of global steps
                          T=1.0,     # Temperature
                          stepsize=0.05, # Smaller step size for perturbations to refine around good solutions
                          seed=seed)

    # 6. Format and return the result (basinhopping returns the best solution found)
    final_params = result.x
    circles = np.zeros((n, 3))
    circles[:, 0] = final_params[:n]       # x-coordinates
    circles[:, 1] = final_params[n:2*n]    # y-coordinates
    circles[:, 2] = final_params[2*n:]     # radii

    return circles

# EVOLVE-BLOCK-END