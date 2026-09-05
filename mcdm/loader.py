import pandas as pd
import numpy as np

def _transform_type_row(type_row: pd.Series) -> list[str]:
    """
    type row 統一轉換成 'benefit'/'cost' 
    支援兩種填法：
        - 數字：0(望大)、1(望小)
        - 文字：'benefit'、'cost'
    """
    criteria_types = []
    for t in type_row:
        if isinstance(t, str):
            label = t.strip().lower()
            if label not in ('benefit', 'cost'):
                criteria_types.append('benefit') #讀不懂直接使用benefit
            criteria_types.append(label)
        else:
            criteria_types.append('cost' if t == 1 else 'benefit')
    return criteria_types


def load_decision_matrix(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """
    讀取決策矩陣Excel,回傳(決策矩陣,準則型態)。
    Excel 格式 ：1.type row 
                2.data matrix
    """
    raw = pd.read_excel(filepath, header=0, index_col=0)
    type_row = raw.loc['type']
    criteria_types = _transform_type_row(type_row)
    matrix = raw.drop('type')
    return matrix, criteria_types


def load_pairwise_matrix(filepath: str) -> dict[str, pd.DataFrame]:
    """
    輸入:
        多位專家：每個工作表對應一位專家，內容為 nxn 準則比較矩陣。
    輸出:
        字典 key=專家, value=準則比較矩陣(pd.DataFrame)
    使用方法：AHP DEMATEL
    """
    sheets = pd.read_excel(filepath, sheet_name=None, header=0, index_col=0)
    return sheets

def load_bwm_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    讀取 BWM 多專家資料，彙整成整體陣列。
    Excel 格式需求：
        - "BO"：每一「列」代表一位專家。
        - "OW"：每一「欄」代表一位專家。
        - BO 的專家人數(列數)須與 OW 的專家人數(欄數)一致，
        - 準則名稱需符合 "C" + 數字 的格式(例如 C1, C2, C3...)。
    輸出:
        (best_idx, worst_idx, BO, OW)
            best_idx: np.ndarray，每位專家 Best 準則的編號(C1→1, C2→2...)，[專家數]
            worst_idx: np.ndarray，每位專家 Worst 準則的編號，[專家數]
            BO: np.ndarray，純數值，每列一位專家的 Best-to-Others 向量，[專家數,準則數]
            OW: np.ndarray，純數值，每欄一位專家的 Others-to-Worst 向量[準則數,專家數]
    供bwm方法使用
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

def load_itara_data(filepath: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    讀取ITARA資料
    Excel 格式需求：
        - 第一列(index='IT')：各準則的無差異閾值(Indifference Threshold)

    輸出:
        matrix: 決策矩陣(不含 IT 列)
        it_values: 每個準則的無差異閾值，長度 = 準則數
    """
    raw = pd.read_excel(filepath, header=0, index_col=0)

    it_values = raw.loc['IT'].to_numpy(dtype=float)
    matrix = raw.drop('IT')

    return matrix, it_values

def load_modified_itara_i_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modified ITARA I專用
    Excel 格式需求：
        -IT：無差異閾值(Indifference Threshold)
        -Aspire Level：各準則的渴望水準
        -Data matrix

    輸出:
        matrix
        it_values: 無差異閾值
        aspire_values:渴望水準
    """
    raw = pd.read_excel(filepath, header=0, index_col=0)

    it_values = raw.loc['IT'].to_numpy(dtype=float)
    aspire_values = raw.loc['Aspire Level'].to_numpy(dtype=float)

    matrix = raw.drop(['IT', 'Aspire Level'])

    return matrix.to_numpy(dtype=float), it_values, aspire_values

def load_modified_itara_ii_data(filepath: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Modified ITARA II專用
    Excel 格式需求：
        -criteria types：
        -IT：無差異閾值(Indifference Threshold)
        -Aspire Level：各準則的渴望水準
        -Data matrix

    輸出:
        -matrix
        -it_values: 無差異閾值
        -aspire_values:渴望水準
        -worst_values:各準則的最差水準
    """
    raw = pd.read_excel(filepath, header=0, index_col=0)

    it_values = raw.loc['IT'].to_numpy(dtype=float)
    aspire_values = raw.loc['Aspire Level'].to_numpy(dtype=float)
    worst_values = raw.loc['Worst Level'].to_numpy(dtype=float)
    
    matrix = raw.drop(['IT', 'Aspire Level','Worst Level'])

    return matrix.to_numpy(dtype=float),it_values, aspire_values,worst_values