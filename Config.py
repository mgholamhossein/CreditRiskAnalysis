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

