"""
all parameters and configurations
"""
DATA_PATH = 'C:\\Users\\user\\ConcordiaCourses\\MachineLearning\\datasets\\sesame_credit_risk\\'
DATA_PATH2 = 'C:\\Users\\user\\ConcordiaCourses\\MachineLearning\\datasets\\sesame_credit_risk\\PreprocessedData-LE\\'
# DATA_PATH2 = 'C:\\Users\\user\\ConcordiaCourses\\MachineLearning\\datasets\\sesame_credit_risk\\PreprocessedData-oneHot\\'

UseTargetEncoding = False
Folds_Num = 5

LGB_PARAM = params = {
    'n_jobs': 4,
    'n_estimators': 10000,
    # 'boosting_type': 'gbdt',
    # 'boost_from_average':'false',
    'learning_rate': 0.014,  # 0.01
    'num_leaves': 56,  # 60,
    # 'num_threads':4,
    'max_depth': -1,
    'tree_learner': "serial",
    'feature_fraction': 0.65,  # 0.58,
    'bagging_freq': 6,  # 5,
    'bagging_fraction': 0.612,  # 0.5,
    'min_data_in_leaf': 110,  # 100,
    'min_child_samples': 25,  # 20,
    'min_child_weight': 12.73,  #
    'silent': -1,
    'verbose': -1,
    'max_bin': 255,
    'bagging_seed': 24  # ,11

}


# {'target': 0.790220215581594, 'params': {'baggingFraction': 0.6120525191312296, 'baggingFreq': 6.0382220168365315,
# 'baggingSeed': 24.115173633512136, 'featureFraction': 0.6499049925956872, 'learnRate': 0.014541399855351297,
# 'maxDepth': 9.469356461614211, 'minChildSample': 24.935617888671082, 'minChildWeight': 12.735718561007117,
# 'minDataInLeaf': 109.83378395711168, 'numLeaves': 56.19020295238769, 'scaleWeight': 3.604479561006932}}





#
# LGB_PARAM = params = {
#     'n_jobs': 4,
#     'n_estimators': 10000,
#     # 'boosting_type': 'gbdt',
#     # 'boost_from_average':'false',
#     'learning_rate': 0.01,
#     'colSam':0.575,
#     'num_leaves': 20,
#     'colSam':0.92,
#     'scaleWeight':0.93,
#     # 'num_threads':4,
#     'max_depth': 3,
#     'tree_learner': "serial",
#     'feature_fraction': 0.6,
#     'bagging_freq': 5,
#     'bagging_fraction': 0.5,
#     'min_data_in_leaf': 100,
#     'min_child_samples': 62,
#     'silent': -1,
#     'verbose': -1,
#     'max_bin': 255,
#     # 'is_unbalance': 'true',
#     'bagging_seed': 11
# }
