import pandas as pd
import numpy as np

def load_decision_matrix(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """
    讀取決策矩陣 Excel,回傳(決策矩陣, 準則型態清單)。
    Excel 格式需求:第一列(index='type')標註每個準則是 0(望大)或 1(望小)。
    """
    raw = pd.read_excel(filepath, header=0, index_col=0)
    type_row = raw.loc['type']
    criteria_types = ['cost' if t == 1 else 'benefit' for t in type_row]
    matrix = raw.drop('type')
    return matrix, criteria_types


def load_pairwise_matrix(filepath: str) -> dict[str, pd.DataFrame]:
    """
    輸入:
        多位專家：每個工作表對應一位專家，內容為 n x n 準則比較矩陣。
    輸出:
        字典，key=專家名稱, value=準則比較矩陣(pd.DataFrame)
    """
    sheets = pd.read_excel(filepath, sheet_name=None, header=0, index_col=0)
    return sheets



def load_bwm_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    讀取 BWM 多專家資料，彙整成整體陣列。
    Excel 格式需求：
        - 工作表 "BO"：每一「列」代表一位專家。
        - 工作表 "OW"：每一「欄」代表一位專家。
        - BO 的專家人數(列數)須與 OW 的專家人數(欄數)一致，
        - 準則名稱需符合 "C" + 數字 的格式(例如 C1, C2, C3...)。
    輸出:
        (best_idx, worst_idx, BO, OW)
            best_idx: np.ndarray，每位專家 Best 準則的編號(C1→1, C2→2...)，形狀 [專家數]
            worst_idx: np.ndarray，每位專家 Worst 準則的編號，形狀 [專家數]
            BO: np.ndarray，形狀(專家數, 準則數)，純數值，每列一位專家的 Best-to-Others 向量
            OW: np.ndarray，形狀(準則數, 專家數)，純數值，每欄一位專家的 Others-to-Worst 向量
    """
    bo_df = pd.read_excel(filepath, sheet_name='BO', header=0, index_col=0)

    ow_raw = pd.read_excel(filepath, sheet_name='OW', header=None)
    worst_labels = ow_raw.iloc[0, 1:].tolist()
    ow_values = ow_raw.iloc[1:, 1:].to_numpy(dtype=float)

    n_experts_bo = len(bo_df)
    n_experts_ow = len(worst_labels)

    if n_experts_bo != n_experts_ow:
        raise ValueError(
            f"BO 表專家數({n_experts_bo})與 OW 表專家數({n_experts_ow})不一致"
        )

    # 從準則名稱字串中取出數字部分，例如 "C1" -> 1
    best_idx = np.array([int(c.replace('C', '')) for c in bo_df.index.tolist()])
    worst_idx = np.array([int(c.replace('C', '')) for c in worst_labels])

    BO = bo_df.to_numpy(dtype=float)
    OW = ow_values

    return best_idx,worst_idx,BO,OW