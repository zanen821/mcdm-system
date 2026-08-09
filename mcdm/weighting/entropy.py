import numpy as np
import pandas as pd


def _invert_cost_criteria(matrix: np.ndarray, criteria_types: list[str]) -> np.ndarray:
    """把成本型(cost)準則做倒數轉換,讓所有欄位統一變成「越大越好」。"""
    transformed = matrix.astype(float).copy()
    for col, ctype in enumerate(criteria_types):
        if ctype == 'cost':
            transformed[:, col] = 1 / transformed[:, col]
    return transformed


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """把每一欄正規化成比例 p_ij,使每欄總和為 1。"""
    col_sum = matrix.sum(axis=0)
    return matrix / col_sum


def _entropy_values(p_matrix: np.ndarray) -> np.ndarray:
    """計算每個準則的熵值 e_j。"""
    k = 1 / np.log(len(p_matrix))
    with np.errstate(divide='ignore', invalid='ignore'):
        plnp = np.where(p_matrix > 0, p_matrix * np.log(p_matrix), 0)
    e_j = -k * plnp.sum(axis=0)
    return e_j


def calculate_weights(matrix: pd.DataFrame, criteria_types: list[str]) -> np.ndarray:
    """
    使用 Entropy Weight Method 計算準則權重。

    輸入:
        matrix: 決策矩陣,列=方案(alternatives),欄=準則(criteria)
        criteria_types: 每個準則是 'benefit'(望大)或 'cost'(望小)

    輸出:
        weights: 每個準則的權重陣列,總和為 1
    """
    data_array = matrix.to_numpy(dtype=float)
    transformed = _invert_cost_criteria(data_array, criteria_types)
    normalized = _normalize(transformed)
    e_j = _entropy_values(normalized)
    d_j = 1 - e_j
    weights = d_j / d_j.sum()
    return weights