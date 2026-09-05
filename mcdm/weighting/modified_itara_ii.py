import numpy as np
import pandas as pd
from mcdm.weighting import modified_itara_i


def _normalize(matrix: np.ndarray,aspire_values: np.ndarray,worst_values: np.ndarray) -> np.ndarray:
    for i in range(len(matrix)):
         for j in range(len(matrix[0])):
            matrix[i][j] = (matrix[i][j]-worst_values[j])/(aspire_values[j]-worst_values[j])
    return matrix

def _best_weights(independent_weight: np.ndarray,dependent_weight: np.ndarray) -> np.ndarray:
    w=np.zeros((2,2),dtype=float)
    w[0][0]=independent_weight@(independent_weight.T)
    w[0][1]=independent_weight@(dependent_weight.T)
    w[1][0]=dependent_weight@(independent_weight.T)
    w[1][1]=dependent_weight@(dependent_weight.T)
    p=np.abs(np.linalg.inv(w)@[(independent_weight@independent_weight.T),dependent_weight@(dependent_weight.T)])
    p=p/np.sum(p)
    weights=np.zeros((len(independent_weight),),dtype=float)
    for i in range(len(independent_weight)):
        weights[i]=p[0]*independent_weight[i]+p[1]*dependent_weight[i]
    return weights
    
def calculate_weights(
    matrix: pd.DataFrame,
    it_values: np.ndarray,
    aspire_values: np.ndarray,
    worst_values: np.ndarray,
    p: float = 2
) -> np.ndarray:
    """
    使用 Modified ITARA 方法計算準則權重。

    輸入:
        -matrix
        -it_values: 無差異閾值(Indifference Threshold)
        -aspire_values
        -worst_values
        -p: 差異度計算的冪次，預設 2
        -alpha: vj 與 cv 的加權比例，預設 0.5(各佔一半)

    輸出:
        weights: 每個準則的權重陣列，總和為 1
    """
    independent_weight=modified_itara_i.independent_weight(matrix, it_values, aspire_values, p)

    matrix=_normalize(matrix,aspire_values,worst_values)
    cv_matrix=np.corrcoef(matrix,rowvar=False)
    dependent_weight=np.sum(1-cv_matrix,axis=1)/np.sum(np.sum(1-cv_matrix,axis=1))
    print(independent_weight)
    print(dependent_weight)
    weights=_best_weights(independent_weight,dependent_weight)
    return weights