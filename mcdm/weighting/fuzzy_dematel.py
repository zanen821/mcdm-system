'''
不可用
'''

from mcdm.aggregation import geometric_mean
import numpy as np
import pandas as pd

def _linguistic_to_fuzzy(matrix: np.ndarray, fuzzy_scale: dict) -> np.ndarray:
    """
    將單一專家的語言評估矩陣(整數代碼)轉換為模糊三角矩陣。
    輸出 shape 為 (n, n, 3)，最後一維依序為 (l, m, u)。
    """
    n = matrix.shape[0]
    fuzzy_matrix = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            fuzzy_matrix[i, j] = fuzzy_scale[matrix[i, j]]
    return fuzzy_matrix


def _aggregate_experts_geometric(
    fuzzy_matrices: dict[str, np.ndarray],
    index=None,
    columns=None,
) -> np.ndarray:
    """
    用既有的 geometric_mean() 聚合多位專家的模糊矩陣。
    對 l, m, u 三個分量分別呼叫 geometric_mean，再組回 (n, n, 3)。

    參數：
        fuzzy_matrices: 字典，key 為專家名稱，value 為 shape (n, n, 3) 的模糊矩陣
        index, columns: 準則名稱，用於建立 DataFrame（若不提供則用預設整數索引）
    """
    n = next(iter(fuzzy_matrices.values())).shape[0]
    if index is None:
        index = range(n)
    if columns is None:
        columns = range(n)

    l_dict = {name: pd.DataFrame(fm[:, :, 0], index=index, columns=columns)
              for name, fm in fuzzy_matrices.items()}
    m_dict = {name: pd.DataFrame(fm[:, :, 1], index=index, columns=columns)
              for name, fm in fuzzy_matrices.items()}
    u_dict = {name: pd.DataFrame(fm[:, :, 2], index=index, columns=columns)
              for name, fm in fuzzy_matrices.items()}

    l_agg = geometric_mean(l_dict).to_numpy()
    m_agg = geometric_mean(m_dict).to_numpy()
    u_agg = geometric_mean(u_dict).to_numpy()

    aggregated = np.stack([l_agg, m_agg, u_agg], axis=-1)
    return aggregated


def _defuzzify(fuzzy_matrix: np.ndarray) -> np.ndarray:
    """
    解模糊化，採用 graded mean integration representation 法：
    crisp = (l + 2m + u) / 4
    fuzzy_matrix shape: (n, n, 3)，回傳 (n, n)
    """
    l = fuzzy_matrix[:, :, 0]
    m = fuzzy_matrix[:, :, 1]
    u = fuzzy_matrix[:, :, 2]
    crisp_matrix = (l + 2 * m + u) / 4
    return crisp_matrix


def fuzzy_dematel(
    expert_matrices: dict[str, np.ndarray],
    fuzzy_scale: dict | None = None,
    index=None,
    columns=None,
) -> np.ndarray:
    """
    Fuzzy DEMATEL 主流程：
    1. 將每位專家的語言評估矩陣轉為模糊三角矩陣
    2. 用 geometric_mean() 聚合所有專家的模糊矩陣（l, m, u 分別做幾何平均）
    3. 解模糊化為明確值矩陣
    4. 對角線歸零 + 正規化
    5. 計算總影響矩陣 T

    參數：
        expert_matrices: 字典，key 為專家名稱，value 為該專家的 n×n 語言評估矩陣，
                          元素為對應 fuzzy_scale 的代碼
        fuzzy_scale: 語言尺度對應表，key 為代碼，value 為 (l, m, u) 三角模糊數。
                     若未提供，使用預設 5 級尺度。
        index, columns: 準則名稱，用於中間 DataFrame 建構

    回傳：
        total_influence_matrix: n×n 總影響矩陣 T
    """
    if fuzzy_scale is None:
        fuzzy_scale = {
            0: (0, 0, 0.25),
            1: (0, 0.25, 0.5),
            2: (0.25, 0.5, 0.75),
            3: (0.5, 0.75, 1),
            4: (0.75, 1, 1),
        }

    fuzzy_matrices = {
        name: _linguistic_to_fuzzy(m, fuzzy_scale)
        for name, m in expert_matrices.items()
    }

    aggregated_fuzzy = _aggregate_experts_geometric(fuzzy_matrices, index=index, columns=columns)
    crisp_matrix = _defuzzify(aggregated_fuzzy)

    np.fill_diagonal(crisp_matrix, 0)
    normalized_matrix = _normalize(crisp_matrix)
    total_influence_matrix = _total_influence(normalized_matrix)

    return total_influence_matrix