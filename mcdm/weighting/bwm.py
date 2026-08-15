import numpy as np
from scipy.optimize import minimize


def _unpack(x: np.ndarray, n_crit: int) -> tuple[np.ndarray, float]:
    """拆分決策變數 x 成權重向量 w 與一致性指標 xi。"""
    w = x[:n_crit]
    xi = x[-1]
    return w, xi


def _build_constraints(
    a_Bj: np.ndarray, a_jW: np.ndarray, best_idx: int, worst_idx: int, n_crit: int
) -> list[dict]:
    """
    建立 BWM 的限制式(對照原始公式)：
        |w_B/w_j - a_Bj| <= xi   for all j   
        |w_j/w_W - a_jW| <= xi   for all j  
        sum(w) = 1
    """
    constraints = []

    for j in range(n_crit):
        # w_B/w_j - a_Bj <= xi
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, j=j: (
                _unpack(x, n_crit)[1]
                - (_unpack(x, n_crit)[0][best_idx] / _unpack(x, n_crit)[0][j] - a_Bj[j])
            )
        })
        # -(w_B/w_j - a_Bj) <= xi
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, j=j: (
                _unpack(x, n_crit)[1]
                + (_unpack(x, n_crit)[0][best_idx] / _unpack(x, n_crit)[0][j] - a_Bj[j])
            )
        })
        # w_j/w_W - a_jW <= xi
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, j=j: (
                _unpack(x, n_crit)[1]
                - (_unpack(x, n_crit)[0][j] / _unpack(x, n_crit)[0][worst_idx] - a_jW[j])
            )
        })
        # -(w_j/w_W - a_jW) <= xi
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, j=j: (
                _unpack(x, n_crit)[1]
                + (_unpack(x, n_crit)[0][j] / _unpack(x, n_crit)[0][worst_idx] - a_jW[j])
            )
        })

    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.sum(_unpack(x, n_crit)[0]) - 1
    })

    return constraints


def _solve_single_expert(
    a_Bj: np.ndarray, a_jW: np.ndarray, best_idx: int, worst_idx: int
) -> tuple[np.ndarray, float]:
    """針對單一專家求解 BWM 非線性模型。"""
    n_crit = len(a_Bj)

    def objective(x):
        _, xi = _unpack(x, n_crit)
        return xi

    constraints = _build_constraints(a_Bj, a_jW, best_idx, worst_idx, n_crit)

    bounds = [(0.001, 1) for _ in range(n_crit)] + [(0, None)]
    x0 = np.array([1 / n_crit] * n_crit + [0.5])

    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1500, 'ftol': 1e-10},
    )

    w_opt, xi_opt = _unpack(result.x, n_crit)
    z=result.fun
    return w_opt, xi_opt,z

def calculate_weights(best_idx: np.ndarray,worst_idx: np.ndarray,BO: np.ndarray,OW: np.ndarray,)->tuple[np.ndarray, np.ndarray]:
    """
    使用 2015 年原始 BWM 方法，針對多位專家分別計算權重。

    輸入:
        best_idx: 每位專家 Best 準則的編號(1-based，例如 C1=1)，形狀 [專家數]
        worst_idx: 每位專家 Worst 準則的編號(1-based)，形狀 [專家數]
        BO: Best-to-Others 矩陣，形狀 [專家數, 準則數]
        OW: Others-to-Worst 矩陣，形狀 [準則數, 專家數]

    輸出:
        weights_all: 形狀為 [專家數, 準則數]，每位專家各自的權重
        xi_all: 形狀為 [專家數]，每位專家的一致性指標 ξ
    """
    n_experts = len(BO)
    n_crit = len(BO[0])

    weights_all = np.zeros((n_experts, n_crit))
    xi_all = np.zeros(n_experts)

    for i in range(n_experts):
        a_Bj = BO[i]
        a_jW = OW[:, i]
        b_idx = best_idx[i] - 1   
        w_idx = worst_idx[i] - 1

        w_opt, xi_opt,z = _solve_single_expert(a_Bj, a_jW, b_idx, w_idx)
        weights_all[i] = w_opt
        xi_all[i] = xi_opt

    average_weights = weights_all.mean(axis=0)

    return weights_all, xi_all, average_weights,z