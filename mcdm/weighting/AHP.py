import numpy as np
import pandas as pd

_RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def _geometric_mean(matrix: np.ndarray) -> np.ndarray:
    """計算準則的幾何平均數"""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(i>j):
                matrix[i][j]=1/matrix[j][i]
            elif(i==j):
                matrix[i][j]=1
            else:
                continue
    arr = np.zeros(matrix.shape[0], dtype=float)
    for i in range(matrix.shape[0]):
        arr[i]=(np.prod(matrix[i,:]))**(1/matrix.shape[0])
    return arr


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """把每一欄正規化成比例 p_ij,使每欄總和為 1 """
    return matrix / matrix.sum()


def _consistency_ratio(matrix: np.ndarray, weights: np.ndarray) -> float:
    """計算一致性比率(Consistency Ratio, CR),< 0.1 代表可接受"""
    n = matrix.shape[0]
    weighted_sum = matrix @ weights
    lambda_vector = weighted_sum / weights
    lambda_max = lambda_vector.mean()

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0.0

    return cr

def calculate_weights(matrix: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    輸入:
        matrix: 決策矩陣,列、欄=準則(criteria)
    輸出:
        weights: 每個準則的權重陣列，總和為 1
        cr: 一致性比率(CR)，< 0.1 代表比較矩陣一致性可接受
    """
    data_array = matrix.to_numpy(dtype=float)
    filled_matrix = data_array.copy()   # 保留原始輸入，避免被 _geometric_mean 內部修改
    geometric_means = _geometric_mean(filled_matrix)
    weights = _normalize(geometric_means)
    cr = _consistency_ratio(filled_matrix, weights)
    return weights,cr