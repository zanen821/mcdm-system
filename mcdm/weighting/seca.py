'''
目前不可用
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _normalize(matrix: np.ndarray, criteria_types: list[str]) -> np.ndarray:
    """
    正規化決策矩陣。
    - 望小(cost)：最小值 / x
    - 望大(benefit)：x / 最大值
    """
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    normalized = np.zeros_like(matrix, dtype=float)
    for col, ctype in enumerate(criteria_types):
        if ctype == 'cost':
            normalized[:, col] = col_min[col] / matrix[:, col]
        else:
            normalized[:, col] = matrix[:, col] / col_max[col]
    return normalized


def _dispersion_and_conflict(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    計算每個準則的「分散程度」(標準差正規化)與「衝突程度」(相關係數轉換)，
    這兩者是 SECA 決定權重時同時考量的依據。
    """
    std_array = np.std(matrix, axis=0)
    std_array = std_array / std_array.sum()

    conflict = 1 - np.corrcoef(matrix, rowvar=False)
    conflict = conflict.sum(axis=1)
    conflict = conflict / conflict.sum()

    return std_array, conflict


def _solve_weights(
    matrix: np.ndarray,
    std_array: np.ndarray,
    conflict_array: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, float, float]:
    """
    用非線性最佳化求解權重，目標是最大化方案間的區別度(lambda_a)，
    同時讓權重不要偏離標準差分佈與衝突程度太多(用 beta 控制懲罰力道)。
    """
    n_alt, n_crit = matrix.shape

    def unpack(x):
        return x[:n_crit], x[-1]

    def objective(x):
        w, lambda_a = unpack(x)
        penalty_std = np.sum((w - std_array) ** 2)
        penalty_conflict = np.sum((w - conflict_array) ** 2)
        return -(lambda_a - beta * (penalty_std + penalty_conflict))

    def make_constraint(i):
        def con(x):
            w, lambda_a = unpack(x)
            s_i = np.dot(matrix[i], w)
            return s_i - lambda_a
        return con

    constraints = [{'type': 'ineq', 'fun': make_constraint(i)} for i in range(n_alt)]
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(unpack(x)[0]) - 1})

    bounds = [(0.001, 1) for _ in range(n_crit)]
    bounds.append((None, None))

    x0 = np.array([1 / n_crit] * n_crit + [0.5])

    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1500, 'ftol': 1e-10},
    )

    w_opt, lambda_a = unpack(result.x)
    return w_opt, lambda_a, -result.fun


def calculate_weights(
    matrix: pd.DataFrame,
    criteria_types: list[str],
    beta: float = 2.0,
) -> np.ndarray:
    """
    使用 SECA 方法計算準則權重（透過非線性最佳化）。

    輸入:
        matrix: 決策矩陣，列=方案，欄=準則
        criteria_types: 每個準則是 'benefit'(望大) 或 'cost'(望小)
        beta: 懲罰係數，控制權重貼近標準差/衝突程度分佈的程度，預設 2

    輸出:
        weights: 每個準則的權重陣列，總和為 1
    """
    data_array = matrix.to_numpy(dtype=float)
    normalized = _normalize(data_array, criteria_types)
    std_array, conflict_array = _dispersion_and_conflict(normalized)
    w_opt, _, _ = _solve_weights(normalized, std_array, conflict_array, beta)
    return w_opt