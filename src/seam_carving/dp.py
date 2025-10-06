import numpy as np

def dp_vertical(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vertical seams (top→bottom).
    Returns (cumulative_costs, steps) where steps[y, x] ∈ {-1, 0, +1} is delta-x.
    """
    energy = energy.astype(np.float64, copy=False)
    h, w = energy.shape
    M = energy.copy()
    steps = np.zeros((h, w), dtype=np.int8)

    for y in range(1, h):
        for x in range(w):
            best_cost = M[y - 1, x]
            best_step = 0
            if x > 0 and M[y - 1, x - 1] < best_cost:
                best_cost = M[y - 1, x - 1]
                best_step = -1
            if x < w - 1 and M[y - 1, x + 1] < best_cost:
                best_cost = M[y - 1, x + 1]
                best_step = +1
            M[y, x] += best_cost
            steps[y, x] = best_step

    return M, steps

def dp_horizontal(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Horizontal seams (left→right).
    Returns (cumulative_costs, steps) where steps[y, x] ∈ {-1, 0, +1} is delta-y.
    """
    energy = energy.astype(np.float64, copy=False)
    h, w = energy.shape
    M = energy.copy()
    steps = np.zeros((h, w), dtype=np.int8)

    for x in range(1, w):
        for y in range(h):
            best_cost = M[y, x - 1]
            best_step = 0
            if y > 0 and M[y - 1, x - 1] < best_cost:
                best_cost = M[y - 1, x - 1]
                best_step = -1
            if y < h - 1 and M[y + 1, x - 1] < best_cost:
                best_cost = M[y + 1, x - 1]
                best_step = +1
            M[y, x] += best_cost
            steps[y, x] = best_step

    return M, steps
