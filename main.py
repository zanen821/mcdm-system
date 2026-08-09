from mcdm.loader import load_decision_matrix
from mcdm.weighting.entropy import calculate_weights
from mcdm.ranking.marcos import rank

matrix,criteria_types = load_decision_matrix('data/entropy_data.xlsx')
weights = calculate_weights(matrix, criteria_types)
print(weights)
result = rank(matrix, weights, criteria_types)
print(result)