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

matrix, criteria_types = load_decision_matrix('data/critic.xlsx')

# 測試不指定 AL
weights = critic.calculate_weights(matrix, criteria_types,[35000000.00,100.00, 90.00, 660000.00, 6.00, 95.00, 1.00, 25.00, 400.00, 15.00, 10.00 ])
print("CRITIC 權重（預設 AL）：")
print(weights)