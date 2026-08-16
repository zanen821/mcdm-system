import numpy as np
from scipy.optimize import linprog


def _build_linear_constraints(a_Bj: np.ndarray, a_jW: np.ndarray, best_idx: int, worst_idx: int, n_crit: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    A_ub @ x <= b_ub
    決策變數 x = [w_1, ..., w_n, xi_L]，共 n_crit + 1 個變數。

    準則 j 產生 4 條不等式(對應絕對值拆開的兩組公式)：
        w_B - a_Bj*w_j - xi_L <= 0
       -w_B + a_Bj*w_j - xi_L <= 0
        w_j - a_jW*w_W - xi_L <= 0
       -w_j + a_jW*w_W - xi_L <= 0
    """
    n_vars = n_crit + 1
    A_ub = []
    b_ub = []

    for j in range(n_crit):
        # w_B - a_Bj*w_j - xi_L <= 0
        row = np.zeros(n_vars)
        row[best_idx] += 1
        row[j] -= a_Bj[j]
        row[-1] = -1
        A_ub.append(row)
        b_ub.append(0)

        # -w_B + a_Bj*w_j - xi_L <= 0
        row = np.zeros(n_vars)
        row[best_idx] -= 1
        row[j] += a_Bj[j]
        row[-1] = -1
        A_ub.append(row)
        b_ub.append(0)

        # w_j - a_jW*w_W - xi_L <= 0
        row = np.zeros(n_vars)
        row[j] += 1
        row[worst_idx] -= a_jW[j]
        row[-1] = -1
        A_ub.append(row)
        b_ub.append(0)

        # -w_j + a_jW*w_W - xi_L <= 0
        row = np.zeros(n_vars)
        row[j] -= 1
        row[worst_idx] += a_jW[j]
        row[-1] = -1
        A_ub.append(row)
        b_ub.append(0)

    return np.array(A_ub), np.array(b_ub)


def _solve_single_expert(
    a_Bj: np.ndarray, a_jW: np.ndarray, best_idx: int, worst_idx: int
) -> tuple[np.ndarray, float, float]:
    """
    求解 2016 線性 BWM 模型。

    輸出:
        w_opt: 最佳權重陣列
        xi_opt: 一致性指標 ξ_L
        z: 求解器目標函數最小值(理論上等於 xi_opt)
    """
    n_crit = len(a_Bj)
    n_vars = n_crit + 1

    # 目標函數：只最小化 xi_L，權重不計入目標
    c = np.zeros(n_vars)
    c[-1] = 1

    A_ub, b_ub = _build_linear_constraints(a_Bj, a_jW, best_idx, worst_idx, n_crit)

    # sum(w) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_crit] = 1
    b_eq = [1]

    bounds = [(0, 1) for _ in range(n_crit)] + [(0, None)]  # w >= 0, xi_L >= 0

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method='highs',
    )

    w_opt = result.x[:n_crit]
    xi_opt = result.x[-1]
    z = result.fun

    return w_opt, xi_opt, z


def calculate_weights(
    best_idx: np.ndarray,
    worst_idx: np.ndarray,
    BO: np.ndarray,
    OW: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2016線性BWM 方法

    輸入:
        best_idx: 每位專家 Best 準則的編號(1-based，例如 C1=1)，形狀 [專家數]
        worst_idx: 每位專家 Worst 準則的編號(1-based)，形狀 [專家數]
        BO: Best-to-Others[專家數, 準則數]
        OW: Others-to-Worst[準則數, 專家數]

    輸出:
        weights_all: 形狀為 [專家數, 準則數]，每位專家各自的權重
        xi_all: 形狀為 [專家數]，每位專家的一致性指標 ξ_L
        z_all: 形狀為 [專家數]，每位專家求解時的目標函數最小值
        average_weights: 形狀為 [準則數]，所有專家權重的算術平均
    """
    n_experts = BO.shape[0]
    n_crit = BO.shape[1]

    weights_all = np.zeros((n_experts, n_crit))
    xi_all = np.zeros(n_experts)
    z_all = np.zeros(n_experts)

    for i in range(n_experts):
        a_Bj = BO[i]
        a_jW = OW[:, i]
        b_idx = best_idx[i] - 1
        w_idx = worst_idx[i] - 1

        w_opt, xi_opt, z = _solve_single_expert(a_Bj, a_jW, b_idx, w_idx)
        weights_all[i] = w_opt
        xi_all[i] = xi_opt
        z_all[i] = z

    average_weights = weights_all.mean(axis=0)

    return weights_all, xi_all, z_all, average_weights