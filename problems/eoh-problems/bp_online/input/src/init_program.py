# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.

    At each step, the item is assigned to the bin with the maximum score.
    Bins whose remaining capacity equals the maximum capacity are excluded
    (they are empty/unused bins). The goal is to minimize the number of used bins.

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Best-fit heuristic: prefer bins where the item fits most tightly,
    # i.e., minimize the remaining space after placing the item.
    remaining = bins - item
    scores = 1.0 / (remaining + 1.0)
    return scores

# EVOLVE-BLOCK-END

