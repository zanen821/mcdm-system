from mcdm.loader import load_pairwise_matrix, load_decision_matrix,load_bwm_data
from mcdm.weighting import ahp,dematel,entropy,2015bwm,2016bwm
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

#2015BWM
best_idx, worst_idx, BO, OW = load_bwm_data('data/temp.xlsx')
weights_all, xi_all, average_weights,z_all= 2015_bwm.calculate_weights(best_idx, worst_idx, BO, OW)

for i in range(len(xi_all)):
    print(f"expert{i+1} 權重：{weights_all[i]} ξ：{xi_all[i]:.4f}" f" z:{z_all[i]:.4f}")
print(f"average weights：{average_weights}")

#2016BWM
best_idx, worst_idx, BO, OW = load_bwm_data('data/temp.xlsx')
weights_all, xi_all, average_weights,z_all= 2016_bwm.calculate_weights(best_idx, worst_idx, BO, OW)

for i in range(len(xi_all)):
    print(f"expert{i+1} 權重：{weights_all[i]} ξ：{xi_all[i]:.4f}" f" z:{z_all[i]:.4f}")
print(f"average weights：{average_weights}")
