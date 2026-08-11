'''
from mcdm.loader import load_decision_matrix
from mcdm.weighting.entropy import calculate_weights
from mcdm.ranking.marcos import rank

matrix,criteria_types = load_decision_matrix('data/entropy_data.xlsx')
weights = calculate_weights(matrix, criteria_types)
print(weights)
result = rank(matrix, weights, criteria_types)
print(result)
'''
# main.py
from mcdm.loader import load_decision_matrix
from mcdm.weighting import entropy, critic

matrix, criteria_types = load_decision_matrix('data/SECA.xlsx')

# 測試不指定 AL
from mcdm.weighting import seca
from mcdm.ranking import weighted_sum

weights = seca.calculate_weights(matrix, criteria_types, beta=2)
print("SECA 權重：", weights)

result = weighted_sum.rank(matrix, weights, criteria_types)
print(result)