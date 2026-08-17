import numpy as np

FUZZY_SCALE = {
    1: (1, 1, 1),
    2: (1, 2, 3),
    3: (2, 3, 4),
    4: (3, 4, 5),
    5: (4, 5, 6),
    6: (5, 6, 7),
    7: (6, 7, 8),
    8: (7, 8, 9),
    9: (8, 9, 9),
}

def to_fuzzy_matrix(matrix: np.ndarray, scale: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把整數比較矩陣(1~9)轉換成三個模糊分量矩陣 (L, M, U)。
    scale 可自訂，不指定則用 FUZZY_SCALE。
    """
    if scale is None:
        scale = FUZZY_SCALE

    L = np.zeros_like(matrix, dtype=float)
    M = np.zeros_like(matrix, dtype=float)
    U = np.zeros_like(matrix, dtype=float)

    it = np.nditer(matrix, flags=['multi_index'])
    for val in it:
        idx = it.multi_index
        l, m, u = scale[int(val)]
        L[idx], M[idx], U[idx] = l, m, u

    return L, M, U