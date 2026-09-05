from mcdm.loader import load_pairwise_matrix, load_decision_matrix,load_bwm_data
from mcdm.weighting import ahp,dematel,entropy,bwm_2015,bwm_2016,fuzzy_bwm
from mcdm.loader import load_decision_matrix
#from mcdm.weighting.entropy import calculate_weights
from mcdm.ranking.marcos import rank

#待修改
#modified ITARA ii
def modified_itara_ii():
    from mcdm.loader import load_modified_itara_ii_data
    from mcdm.weighting.modified_itara_ii import calculate_weights
    matrix,it_values,aspire_values,worst_values= load_modified_itara_ii_data('data/modified_itara_ii.xlsx')
    weights =calculate_weights(matrix,it_values, aspire_values,worst_values)
    print("modified ITARA ii 權重：", weights)

modified_itara_ii()


'''
#modified ITARA i
def modified_itara_i():
    from mcdm.loader import load_modified_itara_i_data
    from mcdm.weighting.modified_itara_i import calculate_weights
    matrix,it_values,aspire_values= load_modified_itara_i_data('data/modified_itara_i.xlsx')
    weights = calculate_weights(matrix, it_values, aspire_values)
    print("modified ITARA i 權重：", weights)

modified_itara_i()
'''


'''
#ITARA
def itara():
    from mcdm.loader import load_itara_data
    from mcdm.weighting.itara import calculate_weights
    matrix,it_values = load_itara_data('data/itara_data.xlsx')
    weights = calculate_weights(matrix, it_values)
    print("ITARA 權重：", weights)

itara()
'''

'''
#DEMATEL 
from mcdm.weighting.dematel import calculate_weights
matrix= load_pairwise_matrix('data/temp.xlsx')
weights,total_impact,net_impact=calculate_weights(matrix)
print("權重：",weights)
print("總影響力：",total_impact)
print("淨影響力：",net_impact)
'''

'''
不可用！！！因為區域求解、全域求解可能造成不同結果
#fuzzy BWM
best_idx, worst_idx, BO, OW = load_bwm_data('data/temp.xlsx')

weights_all, k_all, average_weights = fuzzy_bwm.calculate_weights(
    best_idx, worst_idx, BO, OW
)

print("每位專家權重：", weights_all)
print("每位專家 k*：", k_all)
print("平均權重：", average_weights)
'''

'''
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
#2015BWM
best_idx, worst_idx, BO, OW = load_bwm_data('data/temp.xlsx')
weights_all, xi_all, average_weights,z_all= bwm_2015.calculate_weights(best_idx, worst_idx, BO, OW)

for i in range(len(xi_all)):
    print(f"expert{i+1} 權重：{weights_all[i]} ξ：{xi_all[i]:.4f}" f" z:{z_all[i]:.4f}")
print(f"average weights：{average_weights}")
'''

'''
#2016BWM
best_idx, worst_idx, BO, OW = load_bwm_data('data/temp.xlsx')
weights_all, xi_all, average_weights,z_all= bwm_2016.calculate_weights(best_idx, worst_idx, BO, OW)

for i in range(len(xi_all)):
    print(f"expert{i+1} 權重：{weights_all[i]}  ξ：{xi_all[i]:.4f}" f" z:{z_all[i]:.4f}")
print(f"average weights：{average_weights}")

'''
