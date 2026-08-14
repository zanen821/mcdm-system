import numpy as np
import pandas as pd


def geometric_mean(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    用幾何平均法整合多位專家填寫的矩陣，得出一個共識矩陣。
    輸入:
        matrices: 字典，key 為專家名稱，value 為該專家填的 n x n 矩陣

    輸出:
        整合後的共識矩陣(DataFrame)
    """
    expert_arrays = [m.to_numpy(dtype=float) for m in matrices.values()]
    stacked = np.stack(expert_arrays, axis=0)  # shape: (專家數, n, n)

    n_experts = stacked.shape[0]
    aggregated = np.prod(stacked, axis=0) ** (1 / n_experts)

    first_matrix = next(iter(matrices.values()))
    return pd.DataFrame(aggregated, index=first_matrix.index, columns=first_matrix.columns)


def arithmetic_mean(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    用算術平均法整合多位專家填寫的矩陣，得出一個共識矩陣。
    
    輸入:
        matrices: 字典，key 為專家名稱，value 為該專家填的 n x n 矩陣

    輸出:
        整合後的共識矩陣(DataFrame)
    """
    expert_arrays = [m.to_numpy(dtype=float) for m in matrices.values()]
    stacked = np.stack(expert_arrays, axis=0)  # shape: (專家數, n, n)

    aggregated = np.mean(stacked, axis=0)

    first_matrix = next(iter(matrices.values()))
    return pd.DataFrame(aggregated, index=first_matrix.index, columns=first_matrix.columns)