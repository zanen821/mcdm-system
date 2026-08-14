from mcdm.loader import load_pairwise_matrix, load_decision_matrix,load_bwm_data
from mcdm.weighting.ahp import calculate_weights
from mcdm.weighting.dematel import calculate_weights
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

'''
#AHP CR<0.1可接受
matrix= load_pairwise_matrix('data/AHP.xlsx')
weights,cr = calculate_weights(matrix)
print("AHP 權重：", weights,"CR:",cr)
'''

'''
#DEMATEL 
matrix= load_pairwise_matrix('data/DEMATEL_t.xlsx')
print(calculate_weights(matrix))
'''


data = load_bwm_data('data/BWM_15.xlsx')

print("Best indices:", data['best_idx'])
print("Worst indices:", data['worst_idx'])
print(data['BO'])
print(data['OW'])

