import numpy as np
import pandas as pd
from mcdm import aggregation
#不可用！因為輸出格式不同
def _normalize(matrix: np.ndarray) -> np.ndarray:
    abs_matrix=np.abs(matrix)
    matrix=matrix/max((np.sum(abs_matrix,axis=0)).max(),(np.sum(abs_matrix,axis=1).max()))
    return matrix

def _total_influence(matrix: np.ndarray,alpha:float) -> np.ndarray:
    eigenvalues = np.linalg.eigvals(alpha * matrix)
    spectral_radius = np.max(np.abs(eigenvalues))

    if spectral_radius >= 1:
        raise ValueError(
            f"αM 的頻譜半徑為 {spectral_radius:.4f} ≥ 1，級數不收斂，"
            f"請降低 alpha"
        )
    t=np.linalg.inv(np.eye(len(matrix))-alpha*matrix)
    print(np.eye(len(matrix))-alpha*matrix)
    return  abs(t)


def calculate_weights(pairwise_matrix: pd.DataFrame,alpha=1) -> np.ndarray:
    """
    計算HISA權重
    輸入:
        pairwise_matrix: 成對比較矩陣，DataFrame格式
    輸出:
        weights
    """
    if isinstance(pairwise_matrix, dict):
            pairwise_matrix =aggregation.arithmetic_mean(pairwise_matrix)
    pairwise_matrix = pairwise_matrix.to_numpy(dtype=float)
    matrix=_normalize(pairwise_matrix)
    total_influence=_total_influence(matrix,alpha)
    
    weights=[]
    return weights