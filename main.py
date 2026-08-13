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
from mcdm.loader import load_pairwise_matrix
from mcdm.weighting.AHP import calculate_weights
matrix= load_pairwise_matrix('data/AHP.xlsx')
weights,cr = calculate_weights(matrix)
print("AHP 權重：", weights,"CR:",cr)
'''
