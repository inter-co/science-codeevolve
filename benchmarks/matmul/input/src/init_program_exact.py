# Disable progress bar for cleaner output logs
import os

os.environ["TQDM_DISABLE"] = "1"

# Fixed parameters
n, m, p = 2, 4, 5

# EVOLVE-BLOCK-START
import numpy as np
from dataclasses import dataclass

# This script provides an EXACT solution for the matrix multiplication tensor
# by algorithmically constructing the decomposition for the standard algorithm.
# The rank of this decomposition is n*m*p. The ideia is lower the decomposition rank by using others contructs.


def get_standard_decomposition(
    n: int, m: int, p: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the exact U, V, W factors for the standard matrix
    multiplication algorithm. The rank of this decomposition is n*m*p.
    """
    rank = n * m * p

    # The three factor matrices correspond to selecting A's elements, B's elements,
    # and placing the results in C's elements.
    U = np.zeros((n * m, rank), dtype=np.float32)
    V = np.zeros((m * p, rank), dtype=np.float32)
    W = np.zeros((n * p, rank), dtype=np.float32)

    # Each rank-1 component corresponds to one of the n*m*p scalar multiplications
    # in the standard algorithm.
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This component corresponds to the multiplication A[i,j] * B[j,k]

                # U selects the element A[i,j]
                U[i * m + j, r] = 1

                # V selects the element B[j,k]
                V[j * p + k, r] = 1

                # W places the result in C[i,k]
                # Note: We use the standard k*n+i indexing for the C tensor dimension
                W[k * n + i, r] = 1

                r += 1

    return (U, V, W)


def run():
    """
    Returns the exact decomposition corresponding to the standard algorithm.
    """
    print(f"Generating the exact decomposition for the standard <{n},{m},{p}> algorithm.")

    rank = n * m * p
    decomposition = get_standard_decomposition(n, m, p)

    # This is a perfect solution, so the loss is exactly 0.
    loss = 0.0

    print(f"The rank of this standard algorithm is {rank}.")

    return decomposition, n, m, p, loss, rank


# EVOLVE-BLOCK-END
