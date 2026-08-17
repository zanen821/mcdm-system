import numpy as np
from scipy.optimize import minimize
from mcdm.weighting.scale_conversion import to_fuzzy_matrix
from mcdm.weighting.fuzzy_utils import graded_mean

def _unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """拆分決策變數 x 成 (l, m, u, k)。"""
    l = x[0:n]
    m = x[n:2 * n]
    u = x[2 * n:3 * n]
    k = x[3 * n]
    return l, m, u, k


def _build_constraints(
    a_Bj: tuple[np.ndarray, np.ndarray, np.ndarray],
    a_jW: tuple[np.ndarray, np.ndarray, np.ndarray],
    best_idx: int, worst_idx: int, n: int,
) -> list[dict]:
    """依 Guo & Zhao (2017) 模型，建立 Fuzzy BWM 限制式。"""
    l_Bj, m_Bj, u_Bj = a_Bj
    l_jW, m_jW, u_jW = a_jW
    constraints = []

    for j in range(n):
        # ---- Best 側，l/m/u 三分量各拆兩條 ----
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[0][j]
            - (_unpack(x, n)[0][best_idx] - l_Bj[j] * _unpack(x, n)[0][j])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[0][j]
            + (_unpack(x, n)[0][best_idx] - l_Bj[j] * _unpack(x, n)[0][j])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[1][j]
            - (_unpack(x, n)[1][best_idx] - m_Bj[j] * _unpack(x, n)[1][j])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[1][j]
            + (_unpack(x, n)[1][best_idx] - m_Bj[j] * _unpack(x, n)[1][j])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[2][j]
            - (_unpack(x, n)[2][best_idx] - u_Bj[j] * _unpack(x, n)[2][j])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[2][j]
            + (_unpack(x, n)[2][best_idx] - u_Bj[j] * _unpack(x, n)[2][j])
        )})

        # ---- Worst 側，l/m/u 三分量各拆兩條 ----
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[0][worst_idx]
            - (_unpack(x, n)[0][j] - l_jW[j] * _unpack(x, n)[0][worst_idx])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[0][worst_idx]
            + (_unpack(x, n)[0][j] - l_jW[j] * _unpack(x, n)[0][worst_idx])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[1][worst_idx]
            - (_unpack(x, n)[1][j] - m_jW[j] * _unpack(x, n)[1][worst_idx])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[1][worst_idx]
            + (_unpack(x, n)[1][j] - m_jW[j] * _unpack(x, n)[1][worst_idx])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[2][worst_idx]
            - (_unpack(x, n)[2][j] - u_jW[j] * _unpack(x, n)[2][worst_idx])
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[3] * _unpack(x, n)[2][worst_idx]
            + (_unpack(x, n)[2][j] - u_jW[j] * _unpack(x, n)[2][worst_idx])
        )})

        # ---- 模糊數合法性：l_j <= m_j <= u_j ----
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[1][j] - _unpack(x, n)[0][j]  # m_j - l_j >= 0
        )})
        constraints.append({'type': 'ineq', 'fun': lambda x, j=j: (
            _unpack(x, n)[2][j] - _unpack(x, n)[1][j]  # u_j - m_j >= 0
        )})

    # ---- 正規化：sum(R(w_j)) = 1 ----
    def normalization(x):
        l, m, u, _ = _unpack(x, n)
        return graded_mean(l, m, u).sum() - 1

    constraints.append({'type': 'eq', 'fun': normalization})

    return constraints


def _solve_single_expert(
    a_Bj: tuple[np.ndarray, np.ndarray, np.ndarray],
    a_jW: tuple[np.ndarray, np.ndarray, np.ndarray],
    best_idx: int, worst_idx: int,
) -> tuple[np.ndarray, float]:
    """
    針對單一專家求解 Fuzzy BWM(Guo & Zhao, 2017)。

    輸出:
        weights: 去模糊化並正規化後的最終權重，長度 = 準則數
        k_star: 一致性指標 k*
    """
    n = len(a_Bj[0])
    n_vars = 3 * n + 1

    def objective(x):
        return _unpack(x, n)[3]  # 最小化 k*

    constraints = _build_constraints(a_Bj, a_jW, best_idx, worst_idx, n)

    bounds = [(0.0001, 1)] * (3 * n) + [(0, None)]
    x0 = np.array([1 / n] * (3 * n) + [0.5])

    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-10},
    )

    l, m, u, k_star = _unpack(result.x, n)
    defuzzified = graded_mean(l, m, u)
    weights = defuzzified / defuzzified.sum()

    return weights, k_star


def calculate_weights(
    best_idx: np.ndarray,
    worst_idx: np.ndarray,
    BO: np.ndarray,
    OW: np.ndarray,
    scale: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Fuzzy BWM(Guo & Zhao, 2017)計算多位專家的模糊權重。

    輸入:
        best_idx, worst_idx: 每位專家 Best/Worst 準則編號(1-based)
        BO: Best-to-Others 矩陣(整數 1~9)，形狀 [專家數, 準則數]
        OW: Others-to-Worst 矩陣(整數 1~9)，形狀 [準則數, 專家數]
        scale: 自訂的「數字→模糊數」對照表，不指定則用預設值

    輸出:
        weights_all: 形狀 [專家數, 準則數]，每位專家去模糊化後的權重
        k_all: 形狀 [專家數]，每位專家的一致性指標 k*
        average_weights: 形狀 [準則數]，所有專家權重的算術平均
    """
    BO_l, BO_m, BO_u = to_fuzzy_matrix(BO, scale)
    OW_l, OW_m, OW_u = to_fuzzy_matrix(OW, scale)

    n_experts, n_crit = BO.shape
    weights_all = np.zeros((n_experts, n_crit))
    k_all = np.zeros(n_experts)

    for i in range(n_experts):
        a_Bj = (BO_l[i], BO_m[i], BO_u[i])
        a_jW = (OW_l[:, i], OW_m[:, i], OW_u[:, i])
        b_idx = best_idx[i] - 1
        w_idx = worst_idx[i] - 1

        weights, k_star = _solve_single_expert(a_Bj, a_jW, b_idx, w_idx)
        weights_all[i] = weights
        k_all[i] = k_star

    average_weights = weights_all.mean(axis=0)

    return weights_all, k_all, average_weights