# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

def min_max_dist_dim2_16()->np.ndarray:
    """
    Creates 16 points in 2D to maximize min/max distance ratio.
    This version exploits D4 symmetry (symmetries of a square), assuming the
    optimal configuration consists of one 8-point orbit (generic), one
    4-point orbit on the diagonals, and one 4-point orbit on the axes.
    This reduces the search space from 32 to 4 dimensions.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.
    """
    n_total = 16
    
    def _objective_function_d4_844_axial(params):
        """
        Objective function for the D4 symmetric case (8_gen + 4_diag + 4_axis).
        `params` is a 4-element array: [x, y, d, a] which are coordinates
        of the generators relative to the center (0,0).
        - (x, y): generator for the 8-point orbit.
        - (d, d): generator for the 4-point diagonal orbit.
        - (a, 0): generator for the 4-point axis orbit.
        """
        x, y, d_diag_gen, d_axis_gen = params

        # Generate 8-point orbit from (x, y)
        orbit8 = np.array([
            [x, y], [-y, x], [-x, -y], [y, -x],
            [-x, y], [y, x], [x, -y], [-y, -x]
        ])
        
        # Generate 4-point diagonal orbit from (d, d)
        orbit4_diag = np.array([
            [d_diag_gen, d_diag_gen], [-d_diag_gen, d_diag_gen], 
            [-d_diag_gen, -d_diag_gen], [d_diag_gen, -d_diag_gen]
        ])

        # Generate 4-point axis orbit from (a, 0)
        orbit4_axis = np.array([
            [d_axis_gen, 0], [0, d_axis_gen], 
            [-d_axis_gen, 0], [0, -d_axis_gen]
        ])
        
        points_centered = np.vstack([orbit8, orbit4_diag, orbit4_axis])
        
        # Penalize degenerate configurations that don't produce 16 unique points.
        if np.unique(np.round(points_centered, decimals=8), axis=0).shape[0] != n_total:
            return 1.0 # High penalty for non-unique points

        # Distances are invariant to translation, so we use centered points.
        distances = pdist(points_centered)

        dmin = np.min(distances)
        if dmin < 1e-9: return 1.0

        dmax = np.max(distances)
        if dmax < 1e-9: return 1.0
        
        return -dmin / dmax

    # Bounds for the 4 parameters: [x, y, d_diag, d_axis] relative to center.
    # All coordinates must be within [-0.5, 0.5], so we bound generators to [0, 0.5].
    bounds = [(0, 0.5)] * 4
    np.random.seed(42)

    # Initial guess from known literature for the optimal 16-point maximin design.
    # This provides a very strong starting point near the global optimum.
    # Generators (relative to center in [-0.5, 0.5]^2):
    # g_8:      (0.5, 0.153) -> x=0.5, y=0.153
    # g_4_diag: (0.245, 0.245) -> d=0.245
    # g_4_axis: (0.347, 0) -> a=0.347
    initial_guess = np.array([0.5, 0.153, 0.245, 0.347])

    # DE parameters tuned for this problem
    popsize = 150
    maxiter = 1000
    
    # Initialize population within the bounds and seed with our expert guess.
    init_population = np.random.rand(popsize, 4) * 0.5
    init_population[0] = initial_guess

    result = differential_evolution(
        func=_objective_function_d4_844_axial,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=42,
        disp=False,
        polish=True,
        init=init_population,
        tol=1e-8 # Tighter tolerance for high-precision result
    )

    # --- Reconstruct final solution using optimal parameters ---
    optimal_params = result.x
    center = np.array([0.5, 0.5])
    
    # Unpack the optimal centered generator coordinates
    x, y, d_diag_gen, d_axis_gen = optimal_params
    
    # Reconstruct orbits from centered generators
    orbit8 = np.array([
        [x, y], [-y, x], [-x, -y], [y, -x],
        [-x, y], [y, x], [x, -y], [-y, -x]
    ])
    
    orbit4_diag = np.array([
        [d_diag_gen, d_diag_gen], [-d_diag_gen, d_diag_gen], 
        [-d_diag_gen, -d_diag_gen], [d_diag_gen, -d_diag_gen]
    ])
    
    orbit4_axis = np.array([
        [d_axis_gen, 0], [0, d_axis_gen], 
        [-d_axis_gen, 0], [0, -d_axis_gen]
    ])

    points_centered = np.vstack([orbit8, orbit4_diag, orbit4_axis])
    optimal_points = points_centered + center
    
    return optimal_points
# EVOLVE-BLOCK-END