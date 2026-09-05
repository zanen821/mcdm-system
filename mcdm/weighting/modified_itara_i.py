import numpy as np
import pandas as pd

def _normalize(matrix: np.ndarray, aspire: np.ndarray, it: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    正規化決策矩陣、渴望水準、無差異閾值，並計算每個準則的變異係數(CV)。

    輸出:
        beta_matrix: 正規化後的資料矩陣與渴望水準列合併、排序後的結果
        cv_array: 每個準則的變異係數(標準差 / 平均數)
        nit_array: 正規化後的無差異閾值
    """
    col_sum = matrix.sum(axis=0) + aspire

    normalized_data = matrix / col_sum
    normalized_aspire = aspire / col_sum
    nit_array = it / col_sum

    cv_array = np.std(normalized_data, axis=0, ddof=1) / np.mean(normalized_data, axis=0)

    combined = np.vstack([normalized_data, normalized_aspire])
    beta_matrix = np.sort(combined, axis=0)

    return beta_matrix, cv_array, nit_array


def _gamma(beta_matrix: np.ndarray) -> np.ndarray:
    """計算相鄰排序值之間的差距(gamma)。"""
    n_rows, n_cols = beta_matrix.shape
    gamma_matrix = np.zeros((n_rows - 1, n_cols))
    for row in range(n_rows - 1):
        gamma_matrix[row] = beta_matrix[row + 1] - beta_matrix[row]
    return gamma_matrix


def _delta_and_vj(gamma_matrix: np.ndarray, nit_array: np.ndarray, p: float) -> np.ndarray:
    """
    只保留超過無差異閾值的差距(delta)，並用 p 次方加總後開 p 次方根，得出 vj。
    """
    delta_matrix = np.zeros_like(gamma_matrix)
    n_rows, n_cols = gamma_matrix.shape
    for row in range(n_rows):
        for col in range(n_cols):
            if gamma_matrix[row,col] > nit_array[col]:
                delta_matrix[row,col] = gamma_matrix[row,col] - nit_array[col]

    delta_matrix = delta_matrix ** p
    vj_array = (delta_matrix.sum(axis=0)) ** (1 / p)
    return vj_array


def _combine_weights(vj_array: np.ndarray, cv_array: np.ndarray, alpha: float) -> np.ndarray:
    """結合 vj(差異度)與 cv(變異係數)，用 alpha 做加權合成最終權重。"""
    weights = (
        alpha * (vj_array / vj_array.sum())
        + (1 - alpha) * (cv_array / cv_array.sum())
    )
    return weights

def calculate_weights(
    matrix: pd.DataFrame,
    it_values: np.ndarray,
    aspire_values: np.ndarray,
    p: float = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    使用 Modified ITARA 方法計算準則權重。

    輸入:
        -matrix
        -it_values: 無差異閾值(Indifference Threshold)
        -aspire_values
        -p: 差異度計算的冪次，預設 2
        -alpha: vj 與 cv 的加權比例，預設 0.5(各佔一半)

    輸出:
        weights: 每個準則的權重陣列，總和為 1
    """

    beta_matrix, cv_array, nit_array = _normalize(matrix, aspire_values, it_values)
    gamma_matrix = _gamma(beta_matrix)
    vj_array = _delta_and_vj(gamma_matrix, nit_array, p)
    weights = _combine_weights(vj_array, cv_array, alpha)

    return weights


#供 modified ITARA II 使用
def independent_weight(
    matrix:np.ndarray,
    it_values: np.ndarray,
    aspire_values: np.ndarray,
    p: float = 2) -> np.ndarray:
    beta_matrix,_, nit_array = _normalize(matrix, aspire_values, it_values)
    gamma_matrix= _gamma(beta_matrix)
    vj_array= _delta_and_vj(gamma_matrix, nit_array, p)
    weights=vj_array/vj_array.sum()
    return weights