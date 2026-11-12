"""Seeded spatial samples; distances and bounds use the same units (metres).

Sobol: scipy.stats.qmc scrambled base-2 net. Balance guarantees apply to 2**m
counts; prefixes of other sizes are merely low-discrepancy samples. Poisson-disk
here means sequential uniform dart throwing with a hard exclusion radius, not
Bridson's active-list algorithm or an exact uniform draw over all packings.
"""
import numpy as np
from scipy.stats import qmc


def proposals(bounds, count, seed, method='uniform'):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('bounds_m must contain finite [min,max] pairs')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if method == 'sobol':
        points = qmc.Sobol(d=len(bounds), scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
    elif method in ('uniform', 'poisson_disk'):
        points = np.random.default_rng(seed).random((count, len(bounds)))
    else:
        raise ValueError('method must be uniform, sobol or poisson_disk')
    return qmc.scale(points, bounds[:, 0], bounds[:, 1])


def sample_coordinates(bounds, count, seed, method='uniform', min_distance_m=None):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    if method != 'poisson_disk':
        if min_distance_m is not None:
            raise ValueError('min_distance_m applies only to poisson_disk')
        return proposals(bounds, count, seed, method)
    if isinstance(min_distance_m, bool) or min_distance_m is None or not np.isfinite(min_distance_m) or min_distance_m <= 0:
        raise ValueError('poisson_disk requires positive min_distance_m')
    # 2000 proposals/request is a finite work budget, not a physical constant.
    candidates = proposals(bounds, count * 2000, seed, method)
    result = []
    for point in candidates:
        if all(np.linalg.norm(point-old) >= min_distance_m for old in result):
            result.append(point)
            if len(result) == count:return np.asarray(result)
    raise ValueError('Cannot place requested Poisson-disk samples: reduce count/min_distance_m or enlarge bounds')


def nearest_order(points):
    """Greedy nearest-neighbour visit order; not a shortest-tour guarantee."""
    points = np.asarray(points); remaining = list(range(1, len(points))); order = [0]
    while remaining:
        i = min(remaining, key=lambda j: np.linalg.norm(points[j]-points[order[-1]]))
        order.append(i); remaining.remove(i)
    return points[order]
