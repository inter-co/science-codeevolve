# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements routines for the central voronoi tesselations MAP-Elites algorithm.
#
# ===--------------------------------------------------------------------------------------===#

from typing import List, Tuple

import numpy as np


def closest_centroid_idx(point: np.ndarray, centroids: np.ndarray) -> int:
    """Finds the index of the closest centroid to a given point.
    This function calculates the squared Euclidean distance from the point to each
    centroid and returns the index of the centroid with the minimum distance.
    Args:
        point: A 1D NumPy array representing the coordinates of the point.
        centroids: A 2D NumPy array where each row is a centroid.
    Returns:
        The integer index of the closest centroid in the `centroids` array.
    """
    dist_to_centroids: np.ndarray = np.sum((centroids - point) ** 2, axis=1)
    return np.argmin(dist_to_centroids).item()


def cvt(
    num_centroids: int,
    num_samples: int,
    feature_bounds: List[Tuple[float, float]],
    max_iter: int = 300,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Generates centroids for MAP-Elites using Centroidal Voronoi Tesselation (CVT).
    This function implements Lloyd's algorithm to partition the feature space.
    It works by iteratively sampling points, assigning them to the nearest
    centroid, and updating the centroid to be the mean of its assigned points.
    Args:
        num_centroids: The number of centroids (k) to generate.
        num_samples: The number of random points to sample for partitioning the space.
        feature_bounds: A list of tuples, where each tuple contains the
                        (min_val, max_val) for a feature dimension.
        max_iter: The maximum number of iterations for the algorithm to run.
        tolerance: The convergence threshold. The algorithm stops if the maximum
                   centroid shift between iterations is less than this value.
    Returns:
        A 2D NumPy array of shape (num_centroids, num_features) representing the
        final positions of the centroids.
    """
    num_features = len(feature_bounds)
    samples: np.ndarray = np.array(
        [
            [
                np.random.uniform(feature_bounds[i][0], feature_bounds[i][1])
                for i in range(num_features)
            ]
            for j in range(num_centroids + num_samples)
        ],
        dtype=np.float64,
    )

    centroids: np.ndarray = samples[:num_centroids, :].copy()
    points: np.ndarray = samples[num_centroids : num_centroids + num_samples, :]

    for iteration in range(max_iter):
        prev_centroids: np.ndarray = centroids.copy()

        centroid2points = [[] for _ in range(num_centroids)]
        for i in range(num_samples):
            centroid_idx: int = closest_centroid_idx(points[i, :], centroids)
            centroid2points[centroid_idx].append(i)

        for j in range(num_centroids):
            if centroid2points[j]:
                centroids[j] = np.mean(points[centroid2points[j], :], axis=0)

        centroid_shift: float = np.max(np.linalg.norm(centroids - prev_centroids, axis=1))
        if centroid_shift < tolerance:
            return centroids

    return centroids
