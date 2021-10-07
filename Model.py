import pandas as pd
import numpy as np
import Config
import pickle
import operator
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb  # load lightGBM model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, confusion_matrix, \
    roc_auc_score


# function for target encoding
def target_encoding(tr,va,col):
    """
    tr: trainset which will be use to encode target
    va: validation set which will be merge after caculating target encoding, NOTE that this dataset NEVER been used to
    calculate target
    """
    df = tr[[col,'TARGET']].groupby(col)['TARGET'].agg(['mean'])
    df.columns = ['target_mean_{}'.format(col)]
    va = va.merge(df,on=col,how='left')
    tr = tr.merge(df,on=col,how='left')
    return tr,va


def find_best_threshold(y_lgb, y):
    best_f1_score = -np.inf
    best_thr = 0
    v = [i * 0.01 for i in range(50)]
    for thred in v:
        preds = (y_lgb > thred).astype(int)
        f1 = f1_score(y, preds)
        #     print(thred,f1)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_thr = thred
    return best_thr, best_f1_score


# model features
def model_lightgbm():
    params = Config.LGB_PARAM
    model = lgb.LGBMClassifier(**params)
    return model


def run_model():
    main_df = pd.read_csv(Config.DATA_PATH2 + "aggregated_df.csv")

    # check for some characters
    #     /     ,       -       :
    main_df.columns = main_df.columns.str.replace("/", "_")
    main_df.columns = main_df.columns.str.replace(",", "_")
    main_df.columns = main_df.columns.str.replace("-", "_")
    main_df.columns = main_df.columns.str.replace(":", "_")

    features = [f for f in main_df.columns if f not in ['SK_ID_CURR', 'TARGET']]
    # standardize all features
    # scaler = StandardScaler()

    X = main_df[features]
    y = main_df['TARGET']
    print(X.shape, len(y))

    model_lgb = model_lightgbm()

    X, X_test, y, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=2020, test_size=0.1)
    y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    # (II) 4 using cross validation technique to have 5 models
    num_split = Config.Folds_Num
    kf = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=2020)

    auc_list = []
    recall_list = []
    f1_score_list = []
    models = []

    if Config.UseTargetEncoding:
        X['TARGET'] = y  # to use target encoding, we put target back

    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        X_valid = X.loc[valid_idx]
        y_valid = y.loc[valid_idx]

        if Config.UseTargetEncoding:
            X_train, X_valid = target_encoding(X_train, X_valid, 'ORGANIZATION_TYPE_enc')  # use ORGANIZATION_TYPE to encode target once
            X_train, X_valid = target_encoding(X_train, X_valid, 'OCCUPATION_TYPE_enc')  # use OCCUPATION_TYPE to encode target once
            X_train.drop('TARGET', axis=1, inplace=True)  # after encoding, make sure drop target again
            X_valid.drop('TARGET', axis=1, inplace=True)  # after encoding, make sure drop target again

        model = model_lgb
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc', verbose=300,
                  early_stopping_rounds=300)

        with open(f'sesame_model_test_{i + 1}.pkl', 'wb') as handle:
            pickle.dump(model_lgb, handle)
            X_valid.to_csv(f'Valid_sesame_{i + 1}.pkl.csv', index=False)
            X_train.to_csv(f'Train_sesame_{i + 1}.pkl.csv', index=False)
        handle.close()

        # load pretrained model from saved path
        with open(f'sesame_model_test_{i + 1}.pkl', 'rb') as handle:
            model_lgb = pickle.load(handle)
            y_valid_lgb = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]

            models.append(model_lgb)
            auc_score = roc_auc_score(y_valid, y_valid_lgb)

            best_f1 = -np.inf
            best_thred, best_f1 = find_best_threshold(y_valid_lgb, y_valid)
            print('************************************')
            print('\033[1m' + f'model_{i + 1}:' + '\033[0m')
            print('\033[1m' + 'best_f1,best_thred:\n' + '\033[0m', best_f1, best_thred)

            y_pred_lgb = (y_valid_lgb > best_thred).astype(int)

            print('\033[1m' + f'conf. matrix is: \n' + '\033[0m', confusion_matrix(y_valid, y_pred_lgb))
            print('\033[1m' + 'f1_score: \n' + '\033[0m', f1_score(y_valid, y_pred_lgb))

            auc_list.append(auc_score)

            print('\033[1m' + 'roc_auc_score: \n' + '\033[0m', auc_score)
            f1_score_list.append(f1_score(y_valid, y_pred_lgb))
            print('auc list:', auc_list)

            handle.close()

    # model evaluation using test data
    pred_test = models[0].predict_proba(X_test, num_iteration=models[0].best_iteration_)[:, 1]
    for i in range(1, len(models)):
        pred_test = pred_test + models[i].predict_proba(X_test, num_iteration=models[i].best_iteration_)[:, 1]

    predicted_test = pred_test / num_split

    # best_thred, best_f1 = find_best_threshold(predicted_test, y_test)
    # predicted_test = (predicted_test > best_thred).astype(int)
    print("for test dataset: ", roc_auc_score(y_test, predicted_test))

    print("auc_mean is: ", sum(auc_list)/len(auc_list))
    return models


def check_model_importance():
    with open(f'sesame_model_test_1.pkl', 'rb') as handle:
        model = pickle.load(handle)
        X_train = pd.read_csv('Train_sesame_1.pkl.csv')
        fts = X_train.columns.values  # all features currently used
        len(fts)
        # get feature dictionary with score
        fts_imp = dict(zip(fts, model.feature_importances_))
        fts_imp = sorted(fts_imp.items(), key=operator.itemgetter(1), reverse=True)
        print("fts_imp \n", fts_imp[:100])

        # print feature importance
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X_train.columns})
        plt.figure(figsize=(40, 20))
        sns.set(font_scale=2.5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[:40])
        plt.title('LightGBM Features (for the first model)')  # avg over folds
        plt.tight_layout()
        plt.savefig('lgbm_importances-01.png')
        plt.show()
        handle.close()