import numpy as np
import pandas as pd

def _normlized(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix.sum(axis=0)

def _best_weight(matrix: np.ndarray, expert_weight: np.ndarray) -> np.ndarray:
    matrix=matrix*(expert_weight.reshape(-1,1))
    matrix=matrix/matrix.max(axis=0)
    weights=np.sum(matrix,axis=0)/np.sum(np.sum(matrix,axis=0))
    return weights

def calculate_weight(matrix: np.ndarray,expert_weight: np.ndarray) -> np.ndarray:
    matrix=_normlized(matrix)
    weights=_best_weight(matrix,expert_weight)
    return weights