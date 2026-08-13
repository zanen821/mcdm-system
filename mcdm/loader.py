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

def load_pairwise_matrix(filepath: str) -> pd.DataFrame:
    """
    讀取 AHP 使用的準則兩兩比較矩陣(pairwise comparison matrix)。
    Excel 格式需求：n x n 方陣，列與欄皆為準則名稱。
    """
    matrix = pd.read_excel(filepath, header=0, index_col=0)
    return matrix