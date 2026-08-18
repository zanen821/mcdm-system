import numpy as np
import pandas as pd
from mcdm.aggregation import arithmetic_mean

def _normalize(matrix: np.ndarray) -> np.ndarray:
    """
    正規化：取「列總和」與「欄總和」中的最大值，
    用這個最大值去除整個矩陣。
    """
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)

    row_sums = matrix.sum(axis=1)   
    col_sums = matrix.sum(axis=0)   

    normalized = matrix / max(row_sums.max(), col_sums.max())
    return normalized

def _total_impact(matrix: np.ndarray) -> np.ndarray:
    """
    計算總影響力矩陣 T = D * (I - D)^(-1)
    """
    identity_matrix = np.eye(matrix.shape[0])
    total_influence_matrix = matrix @ np.linalg.inv(identity_matrix - matrix)
    return total_influence_matrix

def weights_impact(total_influence_matrix: np.ndarray) -> np.ndarray:
    """
    計算每個準則的權重，總和為 1
    輸出：
        weight,R+C,R-C
    """
    r_row = total_influence_matrix.sum(axis=1)
    c_col = total_influence_matrix.sum(axis=0)
    weights = (r_row + c_col) / sum(r_row + c_col)
    
    return weights,r_row+c_col,r_row-c_col

def calculate_weights(matrix) -> tuple[np.ndarray, float]:
    """
    輸入:
        matrix: 準則兩兩比較矩陣。可以是：
                - 單一 DataFrame(單一專家)
                - dict[str, DataFrame](多位專家，key 為專家名稱，用算數平均整合)
    輸出:
        weights: 每個準則的權重陣列，總和為 1
    """
    if isinstance(matrix, dict):
        matrix =arithmetic_mean(matrix)

    data_array = matrix.to_numpy(dtype=float)
    normalize = _normalize(data_array)
    total_influence_matrix = _total_impact(normalize)
    weights,total_impact,net_impact= weights_impact(total_influence_matrix)
    return weights,total_impact,net_impact