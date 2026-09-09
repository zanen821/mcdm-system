import numpy as np
import pandas as pd
def _normalize_weights(matrix,criteria_types):
    for i in range(len(criteria_types)):
        if criteria_types[i] == 'cost':
            matrix[:,i] = matrix[:, i]/matrix[:,i].max()
        else:
            matrix[:,i] = matrix[:, i].min()/matrix[:, i]
    return matrix

def _ej(matrix):
    ln_matrix=abs(np.log(matrix))
    sum_ln_matrix=np.sum(ln_matrix,axis=1)
    si=np.zeros(matrix.shape[0])
    si=np.log(1+sum_ln_matrix/matrix.shape[1])

    si_prime=np.zeros((matrix.shape))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            si_prime[i][j]=np.log(1+(-ln_matrix[i][j]+sum_ln_matrix[i])/matrix.shape[1])
    ej=(abs(si_prime-si[:,None])).sum(axis=0)
    return ej

def calculate_weights(matrix:np.ndarray, criteria_types:np.ndarray):
    matrix=_normalize_weights(matrix,criteria_types)
    ej=_ej(matrix)
    weights=ej/ej.sum()
    return weights