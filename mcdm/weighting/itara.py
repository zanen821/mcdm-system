import numpy as np
import pandas as pd


def _normalize(matrix: np.ndarray,it: np.ndarray) -> np.ndarray:
    return matrix / matrix.sum(axis=0),it/ matrix.sum(axis=0)

def _dispersion(matrix:np.ndarray,it) -> np.ndarray:
    matrix=np.sort(matrix,axis=0) 
    dispersion = matrix[1:, :]
    temp=matrix[:-1, :]
    dispersion=dispersion-temp
    dispersion=dispersion-it
    dispersion[dispersion<0]=0
    return dispersion

def _weights(dispersion: np.ndarray,p: float) -> np.ndarray:
    dispersion = np.power(dispersion, p)
    dispersion_sum = np.sum(dispersion, axis=0)
    dispersion_sum=np.power(dispersion_sum,1/p)
    return dispersion_sum / np.sum(dispersion_sum)

def calculate_weights(matrix: pd.DataFrame,it: np.ndarray|None = None,) -> np.ndarray:
    """
    輸入:
        matrix: 決策矩陣,列=方案(alternatives),欄=準則(criteria)
        it:無差異閾值
    輸出:
        weights: 每個準則的權重陣列，總和為 1
    """
    matrix,nit=_normalize(matrix.to_numpy(dtype=float),it)
    dispersion=_dispersion(matrix,nit)
    weights=_weights(dispersion,2)
    return weights
