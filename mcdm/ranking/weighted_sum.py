import numpy as np
import pandas as pd


def rank(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: list[str]) -> pd.DataFrame:
    """
    使用加權總和法(Weighted Sum)進行績效評估。
    這是 SECA 排名步驟所用的方法，但權重可以來自任何權重法
    （Entropy、CRITIC、SECA 皆可），不限定一定要搭配 SECA。

    輸入:
        matrix: 決策矩陣，列=方案，欄=準則（須先正規化，方向統一為望大）
        weights: 每個準則的權重
        criteria_types: 每個準則是 'benefit'(望大) 或 'cost'(望小)

    輸出:
        DataFrame，欄位為 ['alternative', 'score', 'rank']
    """
    data = matrix.to_numpy(dtype=float)
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)

    normalized = np.zeros_like(data)
    for col, ctype in enumerate(criteria_types):
        if ctype == 'cost':
            normalized[:, col] = col_min[col] / data[:, col]
        else:
            normalized[:, col] = data[:, col] / col_max[col]

    scores = normalized @ weights
    ranks = (-scores).argsort().argsort() + 1

    return pd.DataFrame({
        'alternative': matrix.index.tolist(),
        'score': scores,
        'rank': ranks,
    })