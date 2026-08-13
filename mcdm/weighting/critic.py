import numpy as np
import pandas as pd

def _invert_cost_criteria(matrix: np.ndarray, criteria_types: list[str]) -> np.ndarray:
    """把成本型(cost)準則轉望大,統一望大特性"""
    transformed = matrix.astype(float).copy()
    for col, ctype in enumerate(criteria_types):
        if ctype == 'cost':
            transformed[:, col] = 1 / transformed[:, col]
    return transformed


def _normalize_by_aspiration(matrix: np.ndarray, aspire_array: np.ndarray) -> np.ndarray:
    """用渴望水準(aspiration level)對每一欄做正規化"""
    return matrix / aspire_array


def _standard_deviation(matrix: np.ndarray) -> np.ndarray:
    """計算每個準則的標準差,反映該準則的資訊量大小。"""
    return np.std(matrix, axis=0, ddof=1)


def _correlation_weights(matrix: np.ndarray, std_array: np.ndarray) -> np.ndarray:
    """
    用準則間相關係數計算「衝突程度」,結合標準差算出最終權重。
    相關性越低(衝突程度越高)的準則,權重會越高。
    """
    corr = np.corrcoef(matrix, rowvar=False)
    conflict = 1 - corr
    phi = std_array * conflict.sum(axis=0)
    weights = phi / phi.sum()
    return weights

def calculate_weights(matrix: pd.DataFrame,criteria_types: list[str],
    aspire_array: np.ndarray | None = None,) -> np.ndarray:
    """
    輸入:
        matrix: 決策矩陣,列=方案(alternatives),欄=準則(criteria)
        criteria_types: 每個準則是 'benefit'(望大)或 'cost'(望小)
        *aspire_array: 每個準則的渴望水準(AL),可選填
                      不指定時，預設用每個準則轉換後的最大值代替。
    輸出:
        weights: 每個準則的權重陣列，總和為 1
    """
    data_array = matrix.to_numpy(dtype=float)
    transformed = _invert_cost_criteria(data_array, criteria_types)

    if aspire_array is None:
        aspire_array = transformed.max(axis=0)

    normalized = _normalize_by_aspiration(transformed, aspire_array)
    std_array = _standard_deviation(normalized)
    weights = _correlation_weights(normalized, std_array)
    return weights