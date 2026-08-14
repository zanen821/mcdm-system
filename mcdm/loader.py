import pandas as pd

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