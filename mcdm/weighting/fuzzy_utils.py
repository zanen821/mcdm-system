import numpy as np

def graded_mean(l: np.ndarray, m: np.ndarray, u: np.ndarray) -> np.ndarray:
    """R(w) = (l + 4m + u) / 6"""
    return (l + 4 * m + u) / 6