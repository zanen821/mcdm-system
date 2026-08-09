import numpy as np
import pandas as pd

def _reference_values(data:np.ndarray,criteria_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    計算每個準則的理想解AI與反理想解AAI
    - 望大(benefit):理想解取最大值,反理想解取最小值
    - 望小(cost):理想解取最小值,反理想解取最大值
    """
    n_criteria = data.shape[1]#取準則數
    ideal=np.zeros(n_criteria) 
    anti_ideal = np.zeros(n_criteria)
    for col in range(n_criteria):
        if criteria_types[col] == 'cost':
            ideal[col] = data[:, col].min()
            anti_ideal[col] = data[:, col].max()
        else:  # benefit
            ideal[col] = data[:, col].max()
            anti_ideal[col] = data[:, col].min()
    return ideal, anti_ideal

def _normalize(value: np.ndarray, ideal: np.ndarray, criteria_types: list[str]) -> np.ndarray:
    """
    以理想解為基準做正規化。
    - 望大:x / 理想解
    - 望小:理想解 / x
    """
    normalized = np.zeros_like(value,dtype=float)
    for col in range(len(criteria_types)):
        if criteria_types[col] == 'cost':
            normalized[..., col] = ideal[col] / value[..., col]
        else:
            normalized[..., col] = value[..., col] / ideal[col]
    return normalized

def _utility_function(k_neg: np.ndarray, k_pos: np.ndarray) -> np.ndarray:
    """MARCOS 的效用函數 f(Ki),綜合考量與理想解、反理想解的效用度 """
    ki = k_neg + k_pos
    return ki / (1 + (1 - k_pos / ki) / (k_pos / ki) + (1 - k_neg / ki) / (k_neg / ki))


def rank(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: list[str]) -> pd.DataFrame:
    """
    使用 MARCOS 進行績效評估。

    輸入:
        matrix: 決策矩陣,列=方案(alternatives),欄=準則(criteria)
        weights: 每個準則的權重
        criteria_types: 每個準則是 'benefit'(望大)或 'cost'(望小)

    輸出:
        DataFrame,欄位為 ['alternative', 'score', 'rank']
    """
    data = matrix.to_numpy(dtype=float)
    ideal, anti_ideal = _reference_values(data, criteria_types)

    normalized_data = _normalize(data, ideal, criteria_types)
    normalized_ideal = _normalize(ideal.reshape(1, -1), ideal, criteria_types).flatten()
    normalized_anti_ideal = _normalize(anti_ideal.reshape(1, -1), ideal, criteria_types).flatten()

    weighted_data = normalized_data * weights
    weighted_ideal = normalized_ideal * weights
    weighted_anti_ideal = normalized_anti_ideal * weights

    si = weighted_data.sum(axis=1)
    ideal_sum = weighted_ideal.sum()
    anti_ideal_sum = weighted_anti_ideal.sum()

    k_neg = si / anti_ideal_sum
    k_pos = si / ideal_sum

    scores = _utility_function(k_neg, k_pos)
    ranks = (-scores).argsort().argsort() + 1

    return pd.DataFrame({
        'alternative': matrix.index.tolist(),
        'score': scores,
        'rank': ranks,
    })