# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    Args:
        current_sequence: list of job indices in the current sequence.
        time_matrix: np.ndarray of shape (n, m) - execution time of each job
            on each machine.
        m: int - number of machines.
        n: int - number of jobs.

    Returns:
        new_matrix: np.ndarray of shape (n, m) - updated execution time matrix.
        perturb_jobs: list of job indices to perturb (length > 1, max 5).
    """
    new_matrix = time_matrix.copy()

    # Penalize jobs on the critical path by increasing their processing times
    total_times = np.sum(time_matrix, axis=1)
    max_time = np.max(total_times)
    for i in range(n):
        ratio = total_times[i] / max_time
        new_matrix[i] = time_matrix[i] * (1.0 + 0.1 * ratio)

    # Select top jobs by total processing time as perturbation targets
    perturb_jobs = list(np.argsort(-total_times)[:5])

    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
