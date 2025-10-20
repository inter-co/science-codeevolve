# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
import os
import math
import itertools # For detailed post-validation overlap checks

def circle_packing21()->np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 to maximize the sum of their radii.
    Uses a parallelized, multi-stage optimization approach with diversified structured initial guesses.
    """
    N_CIRCLES = 21
    MASTER_SEED = 42
    N_TRIALS = min(os.cpu_count() or 1, 32)
    PERIMETER = 4.0
    RADIUS_LOWER_BOUND = 1e-9
    MIN_RECT_DIM = 0.05
    MAX_RECT_DIM = PERIMETER / 2.0 - MIN_RECT_DIM
    # Upper bound for any single circle's radius. Max is 0.5 in a 1x1 square.
    # Using a slightly smaller value for numerical stability.
    MAX_RADIUS_BOUND = (PERIMETER / 4.0) - RADIUS_LOWER_BOUND * 2 # Effectively 0.5 - buffer

    def _generate_initial_params_s1(run_idx: int, local_rng: np.random.Generator):
        """
        Generates a smart, feasible initial guess for Stage 1 (uniform radius optimization),
        including diversified rectangle aspect ratios and different circle placement patterns.
        Combines grid-based and phyllotaxis patterns from Inspiration 3.
        """
        # 1. Diversify initial rectangle aspect ratios for broad exploration
        initial_widths_options = [1.0, 1.61803, 0.61803, 1.25, 0.75, 1.5, 0.5, 1.1, 0.9, 1.3, 0.8]
        initial_width = initial_widths_options[run_idx % len(initial_widths_options)]
        initial_width += local_rng.uniform(-0.05, 0.05)
        initial_width = np.clip(initial_width, MIN_RECT_DIM, MAX_RECT_DIM)
        initial_height = PERIMETER / 2.0 - initial_width

        # 2. Generate initial circle centers using one of two strategies
        if run_idx % 2 == 0:  # Strategy 1: Perturbed Grid (from Inspiration 1/Target)
            rows_base = int(math.ceil(math.sqrt(N_CIRCLES)))
            cols_base = int(math.ceil(N_CIRCLES / rows_base))
            # Diversify grid orientation and density slightly
            rows, cols = (rows_base, cols_base) if run_idx % 4 < 2 else (cols_base, rows_base)

            grid_margin = 0.1
            grid_x_coords = np.linspace(grid_margin, 1.0 - grid_margin, cols)
            grid_y_coords = np.linspace(grid_margin, 1.0 - grid_margin, rows)
            
            centers_norm = np.array([(gx, gy) for gy in grid_y_coords for gx in grid_x_coords])[:N_CIRCLES]
            
            initial_centers = centers_norm * np.array([initial_width, initial_height])
            
            jitter_scale = 0.05
            jitter = local_rng.uniform(-1, 1, initial_centers.shape) * np.array([initial_width, initial_height]) * jitter_scale
            initial_centers += jitter

        else:  # Strategy 2: Phyllotaxis (Sunflower) for more isotropic initial layouts (from Inspiration 3)
            golden_angle = np.pi * (3. - np.sqrt(5.))
            indices = np.arange(N_CIRCLES) + 0.5 # Use 0.5 offset for better center point
            
            radius_norm = np.sqrt(indices / N_CIRCLES)
            theta = golden_angle * indices
            
            # Scale to fit ellipse within the rectangle with a small margin
            margin_factor = 0.95
            x_centers = (initial_width / 2.0) + (initial_width / 2.0) * margin_factor * radius_norm * np.cos(theta)
            y_centers = (initial_height / 2.0) + (initial_height / 2.0) * margin_factor * radius_norm * np.sin(theta)
            
            initial_centers = np.column_stack((x_centers, y_centers))

        # 3. Common post-processing for all strategies: Clip centers to be within bounds
        # Use a slightly smaller buffer for clipping initial centers to allow more flexibility
        center_clip_buffer = RADIUS_LOWER_BOUND * 10 # Adjusted buffer
        initial_centers[:, 0] = np.clip(initial_centers[:, 0], center_clip_buffer, initial_width - center_clip_buffer)
        initial_centers[:, 1] = np.clip(initial_centers[:, 1], center_clip_buffer, initial_height - center_clip_buffer)

        # 4. Calculate a feasible initial uniform radius (vectorized and robust)
        # Containment limits for uniform radius (vectorized for clarity and speed)
        max_r_contain = min(
            np.min(initial_centers[:, 0]),
            np.min(initial_width - initial_centers[:, 0]),
            np.min(initial_centers[:, 1]),
            np.min(initial_height - initial_centers[:, 1])
        )
        max_r_uniform = max_r_contain

        # Non-overlap limits for uniform radius
        if N_CIRCLES > 1:
            dist_sq = pdist(initial_centers, 'sqeuclidean')
            if dist_sq.size > 0:
                max_r_uniform = min(max_r_uniform, np.sqrt(np.min(dist_sq)) / 2.0)

        # 5. Finalize initial radius with safety buffers
        initial_radius = max(RADIUS_LOWER_BOUND, max_r_uniform - RADIUS_LOWER_BOUND * 10) # Small buffer
        initial_radius = min(initial_radius, min(initial_width, initial_height) / 2.0 - RADIUS_LOWER_BOUND)
        initial_radius = min(initial_radius, MAX_RADIUS_BOUND) # Ensure it doesn't exceed global max radius

        return initial_radius, initial_width, initial_height, initial_centers

    # --- Stage 2/3/4 shared functions, refactored for reusability (from Inspiration 3) ---
    def _extract_sX(v):
        # Assumes v is [width, x1, y1, r1, x2, y2, r2, ...]
        w = v[0]
        h = PERIMETER / 2.0 - w
        circles_flat = v[1:].reshape((N_CIRCLES, 3))
        c, r = circles_flat[:, :2], circles_flat[:, 2]
        return w, h, c, r

    def _obj_sX(v):
        _, _, _, r = _extract_sX(v)
        return -np.sum(r)

    def _con_contain_sX(v):
        w, h, c, r = _extract_sX(v)
        return np.concatenate([
            c[:, 0] - r,                      # x - r >= 0
            w - c[:, 0] - r,                  # w - x - r >= 0
            c[:, 1] - r,                      # y - r >= 0
            h - c[:, 1] - r                   # h - y - r >= 0
        ])

    def _con_overlap_sX(v):
        _, _, c, r = _extract_sX(v)
        if N_CIRCLES <= 1: return np.array([]) # Handle single circle case (from Inspiration 3)
        
        dist_sq = pdist(c, 'sqeuclidean')
        indices = np.triu_indices(N_CIRCLES, k=1)
        radii_sums_sq = (r[indices[0]] + r[indices[1]])**2
        return dist_sq - radii_sums_sq # Must be >= 0 for no overlap

    def _con_positive_radii_sX(v):
        _, _, _, r = _extract_sX(v)
        return r - RADIUS_LOWER_BOUND # Must be >= 0

    def _run_multi_stage_optimization(seed: int):
        local_rng = np.random.default_rng(seed)
        
        # --- Stage 1: Uniform Radius Optimization (SLSQP) ---
        # Objective to maximize uniform radius 'r' (minimize -r)
        def _extract_s1(v):
            r, w = v[0], v[1]
            h = PERIMETER / 2.0 - w
            c = v[2:].reshape((N_CIRCLES, 2))
            return r, w, h, c
        def _obj_s1(v): return -v[0]
        def _con_contain_s1(v):
            r, w, h, c = _extract_s1(v); return np.concatenate([c[:, 0] - r, w - c[:, 0] - r, c[:, 1] - r, h - c[:, 1] - r])
        def _con_overlap_s1(v):
            r, _, _, c = _extract_s1(v)
            if N_CIRCLES <= 1: return np.array([])
            dist_sq = pdist(c, 'sqeuclidean'); return dist_sq - (2 * r)**2

        r0_s1, w0_s1, h0_s1, c0_s1 = _generate_initial_params_s1(seed, local_rng)
        x0_s1 = np.concatenate([np.array([r0_s1, w0_s1]), c0_s1.flatten()])
        bounds_s1 = [(RADIUS_LOWER_BOUND, MAX_RADIUS_BOUND)] + [(MIN_RECT_DIM, MAX_RECT_DIM)] + [(0.0, MAX_RECT_DIM)] * (N_CIRCLES * 2)
        cons_s1 = [{'type': 'ineq', 'fun': _con_contain_s1}, {'type': 'ineq', 'fun': _con_overlap_s1}]
        
        res_s1 = minimize(_obj_s1, x0_s1, method='SLSQP', bounds=bounds_s1, constraints=cons_s1,
                          options={'maxiter': 5000, 'disp': False, 'ftol': 1e-9})

        if not res_s1.success:
            # Return a dummy result for consistency (from Inspiration 3)
            return type('obj', (object,), {'success': False, 'fun': float('inf'), 'x': np.zeros(1 + 1 + N_CIRCLES*2)})()

        r1, w1, h1, c1 = _extract_s1(res_s1.x)

        # --- Stage 2: Radii Polishing Step (SLSQP - optimize radii only, fixed centers/width) ---
        def _polish_obj(r_vec): return -np.sum(r_vec)
        def _polish_con_contain(r_vec):
            return np.concatenate([c1[:, 0] - r_vec, w1 - c1[:, 0] - r_vec, c1[:, 1] - r_vec, h1 - c1[:, 1] - r_vec])
        def _polish_con_overlap(r_vec):
            dist_sq = pdist(c1, 'sqeuclidean')
            indices = np.triu_indices(N_CIRCLES, k=1)
            radii_sums_sq = (r_vec[indices[0]] + r_vec[indices[1]])**2
            return dist_sq - radii_sums_sq
        
        polish_bounds = [(RADIUS_LOWER_BOUND, MAX_RADIUS_BOUND)] * N_CIRCLES
        polish_cons = [{'type': 'ineq', 'fun': _polish_con_contain}, {'type': 'ineq', 'fun': _polish_con_overlap}]
        
        res_polish = minimize(_polish_obj, np.full(N_CIRCLES, r1), method='SLSQP', bounds=polish_bounds, constraints=polish_cons,
                              options={'maxiter': 1000, 'disp': False, 'ftol': 1e-9})
        
        r_polished = res_polish.x if res_polish.success else np.full(N_CIRCLES, r1)

        # --- Stage 3: Full Variable Optimization (SLSQP - warm start from polished radii) ---
        x0_s3 = np.concatenate([np.array([w1]), np.hstack([c1, r_polished.reshape(N_CIRCLES, 1)]).flatten()])
        
        bounds_s3 = [(MIN_RECT_DIM, MAX_RECT_DIM)]
        for _ in range(N_CIRCLES):
            bounds_s3.extend([(0.0, MAX_RECT_DIM), (0.0, MAX_RECT_DIM), (RADIUS_LOWER_BOUND, MAX_RADIUS_BOUND)])
        
        cons_s3_slsqp = [{'type': 'ineq', 'fun': _con_contain_sX},
                         {'type': 'ineq', 'fun': _con_overlap_sX},
                         {'type': 'ineq', 'fun': _con_positive_radii_sX}]

        res_s3 = minimize(_obj_sX, x0_s3, method='SLSQP', bounds=bounds_s3, constraints=cons_s3_slsqp,
                          options={'maxiter': 8000, 'disp': False, 'ftol': 1e-9}) # Reduced maxiter, adjusted ftol
        
        if not res_s3.success: return res_s3 # Return early if Stage 3 fails

        # --- Stage 4: Refinement with trust-constr (from Inspiration 3) ---
        # Convert SLSQP constraints to NonlinearConstraint objects for trust-constr
        cons_s4_nonlinear = [
            NonlinearConstraint(_con_contain_sX, 0, np.inf),
            NonlinearConstraint(_con_overlap_sX, 0, np.inf),
            NonlinearConstraint(_con_positive_radii_sX, 0, np.inf)
        ]
        
        # Use Stage 3's best result as warm start for trust-constr
        res_s4 = minimize(_obj_sX, res_s3.x, method='trust-constr', bounds=bounds_s3, constraints=cons_s4_nonlinear,
                          options={'maxiter': 3000, 'verbose': 0, 'gtol': 1e-10}) # Reduced maxiter for refinement

        # If trust-constr fails or yields a worse result, revert to SLSQP's best.
        final_res = res_s4 if res_s4.success and res_s4.fun < res_s3.fun else res_s3

        # --- Post-optimization feasibility check (from Inspiration Program 1 and 3) ---
        # Even if final_res.success is true, actual violations might exist due to ftol/gtol.
        if final_res.success:
            current_violations = np.concatenate([
                _con_contain_sX(final_res.x),
                _con_overlap_sX(final_res.x),
                _con_positive_radii_sX(final_res.x)
            ])
            # Sum up all negative violations (actual constraint breaches)
            total_violation_amount = np.sum(np.maximum(0, -current_violations))
            
            # Only consider the result truly successful if violations are negligible
            if total_violation_amount > 1e-7: 
                final_res.success = False # Mark as unsuccessful if significant violations

        return final_res

    # --- Main Multi-start Logic ---
    rng = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(low=0, high=2**31, size=N_TRIALS)

    print(f"Starting {N_TRIALS} parallel multi-stage optimization trials...")
    results = Parallel(n_jobs=-1, backend='loky')(delayed(_run_multi_stage_optimization)(seed) for seed in seeds) # Explicitly set backend='loky' for robustness

    successful_results = [res for res in results if res.success]
    if not successful_results:
        print("Warning: No optimization trial converged successfully. Returning the best result found (may be infeasible).")
        best_result = min(results, key=lambda res: res.fun if hasattr(res, 'fun') else float('inf'))
    else:
        best_result = min(successful_results, key=lambda res: res.fun)

    if not best_result.success:
        print("\nFATAL: Best result did not converge successfully. Returning an array of zeros.")
        return np.zeros((N_CIRCLES,3))

    optimized_width = best_result.x[0]
    optimized_height = PERIMETER / 2.0 - optimized_width
    final_circles = best_result.x[1:].reshape((N_CIRCLES, 3))

    print(f"Best result found with sum_radii = {-best_result.fun:.15f}")
    if not best_result.success:
        print(f"Best run's status: {best_result.message}")
    print(f"Optimized container dimensions: Width={optimized_width:.4f}, Height={optimized_height:.4f}")

    # --- Comprehensive Post-optimization Validation (from Inspiration 3) ---
    TOL = 1e-7 # A small tolerance for final validation checks

    # Check positive radii
    if np.any(final_circles[:, 2] < RADIUS_LOWER_BOUND - TOL):
        print(f"WARNING: Some radii are not positive (post-validation). Min radius: {np.min(final_circles[:, 2]):.2e}")

    # Check containment
    containment_violations = np.concatenate([
        final_circles[:, 0] - final_circles[:, 2],                          # x_min - r >= 0
        optimized_width - final_circles[:, 0] - final_circles[:, 2],      # w - x_max - r >= 0
        final_circles[:, 1] - final_circles[:, 2],                          # y_min - r >= 0
        optimized_height - final_circles[:, 1] - final_circles[:, 2]      # h - y_max - r >= 0
    ])
    if np.any(containment_violations < -TOL):
        print(f"WARNING: Some circles are not contained (post-validation). Max violation: {np.max(-containment_violations):.2e}")

    # Check non-overlap
    centers = final_circles[:, :2]
    radii = final_circles[:, 2]
    if N_CIRCLES > 1:
        center_distances_sq = pdist(centers, metric='sqeuclidean')
        indices_rows, indices_cols = np.triu_indices(N_CIRCLES, k=1)
        radii_sums_sq = (radii[indices_rows] + radii[indices_cols])**2
        
        overlap_violations = radii_sums_sq - center_distances_sq # Positive means overlap
        if np.any(overlap_violations > TOL):
            print(f"WARNING: Some circles are overlapping (post-validation). Max violation: {np.max(overlap_violations):.2e}")
            # Optional: detailed printing of overlaps, as in Inspiration 3
            # for k, (i, j) in enumerate(itertools.combinations(range(N_CIRCLES), 2)):
            #     if overlap_violations[k] > TOL:
            #         dist = np.sqrt(center_distances_sq[k])
            #         r_sum = radii[i] + radii[j]
            #         print(f"  Overlap between circles {i} and {j}: dist={dist:.8f}, r_sum={r_sum:.8f}, violation={r_sum - dist:.8f}")

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == '__main__':
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
