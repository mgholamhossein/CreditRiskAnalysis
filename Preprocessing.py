import numpy as np  # import numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd  # import pandas


def preprocessing(df, df_num=0):
    # list of features (cols) with NAN values
    cols_with_missing = df.loc[:, df.isnull().any()].columns
    # create a data frame for the features with the missing values if exist any
    if len(cols_with_missing) != 0:
        df_miss = pd.DataFrame(cols_with_missing, columns=['Feature_Name']).reset_index()
        # add the number of missing values for each feature and the percentage of missing data for each feature
        for f in range(len(cols_with_missing)):
            df_miss.loc[f, 'nan_count'] = df[df_miss.Feature_Name[f]].isnull().sum()
            df_miss.loc[f, 'nan_percentage'] = df_miss.loc[f, 'nan_count'] / len(df)

        df_miss.sort_values(by='nan_count', ascending=False, inplace=True)
        df_miss = df_miss.set_index('index')

        # Drop the features that have more than 90% missing values if there is any
        columns_remove = df_miss[df_miss.nan_percentage > 0.9].Feature_Name.tolist()
        df.drop(columns_remove, 1, inplace=True)

    # Indicate the list features whose missing values should be replaced by:
    # unknown, mean or median or -1 for each dataset
    replace_unknown_list = replace_median_list = replace_mean_list = replace_minus_one_list = []
    if df_num == 0:   # train_df, application_train.csv
        replace_mean_list = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                             'COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG',  'LANDAREA_AVG',
         'LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE',
         'BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE',
         'ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE',
         'NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
         'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI',
         'LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI',
         'TOTALAREA_MODE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
        replace_median_list = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS',
                               'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR', \
                               'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                               'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', \
                               'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                               'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_EMPLOYED']
        replace_unknown_list = ['NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
                                'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

    elif df_num == 1: # bureau_df, bureau.csv
        replace_median_list = ['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM',
                               'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_ANNUITY']
    elif df_num == 2: # bureau_balance_df, bureau_balance.csv
        print("bureau_balance_df doesn't have any missing value")
    elif df_num == 3: # credit_balance_df, credit_card_balance.csv
        replace_median_list = ['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT', \
                               'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', \
                                'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT','CNT_INSTALMENT_MATURE_CUM']
    elif df_num == 5: # installments_df, installments_payments.csv
        replace_median_list = ['DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT']
    elif df_num == 6: # POS_CASH_balance_df, POS_CASH_balance.csv
        replace_median_list = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
    elif df_num == 7: # prev_app_df, previous_application.csv
        replace_median_list = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'CNT_PAYMENT',
                               'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', \
                               'DAYS_TERMINATION']

        replace_mean_list = ['RATE_DOWN_PAYMENT']

        replace_unknown_list = ['NAME_TYPE_SUITE', 'PRODUCT_COMBINATION']
        replace_minus_one_list = ['NFLAG_INSURED_ON_APPROVAL']
    else:
        print("wrong dataframe number")

    # Fill the nan values:
    if len(replace_unknown_list) != 0:
        for Cat_feat in replace_unknown_list:
            df[Cat_feat].fillna('unknown', inplace=True)
            # print(Cat_feat, df[Cat_feat].unique())

    if len (replace_mean_list) != 0:
        for Num_feat in replace_mean_list:
            df[Num_feat].fillna(df[Num_feat].mean (), inplace=True)
            # print(Num_feat, df[Num_feat].isnull().sum())

    if len (replace_median_list) != 0:
        for Num_feat in replace_median_list:
            df[Num_feat].fillna(df[Num_feat].median (), inplace=True)
            # print(Num_feat, df[Num_feat].isnull().sum())

    if len(replace_minus_one_list) != 0:
        for Num_feat in replace_minus_one_list:
            df[Num_feat].fillna(-1, inplace=True)
            # print(Num_feat, df[Num_feat].unique())

    # process categorical
    # First create the list we want to one-hot encode and label encode
    cat_cols = [f for f in df if 'object' in str (df[f].dtype)]
    if len(cat_cols) == 0:
        print("file number %d doesn't have any categorical feature"%df_num)
    one_hot_list = []
    label_encode_list = cat_cols
    # for f in cat_cols:
    #     if df[f].nunique() > 3:
    #         one_hot_list.append(f)
    #     else:
    #         label_encode_list.append(f)

    # Label Encoding:
    le_list = [f for f in df.columns.values if f in label_encode_list]
    for f in le_list:
        col_names = []
        s = df[f].unique()
        col_names.append(f + "_enc")

        le = LabelEncoder()
        feat_cat = df[f]
        feat_encoded = le.fit_transform(feat_cat)
        features_df = pd.DataFrame(feat_encoded, columns=col_names)
        df = pd.concat ([df, features_df], axis=1, verify_integrity=True)

    # One Hot encoding:
    ohe_list = [f for f in df.columns.values if f in one_hot_list]
    for f in ohe_list:
        col_names = []
        s = df[f].unique()

        for i in range(len(s)):
            s[i] = s[i].strip()
            col_names.append(f + "_" + s[i].replace(" ", "_"))

        binary_encoder = OneHotEncoder(categories='auto')
        feat_1hot = binary_encoder.fit_transform(df[f].values.reshape(-1, 1))
        feat_1hot_mat = feat_1hot.toarray()
        feat_df = pd.DataFrame(feat_1hot_mat, columns=col_names)

        df = pd.concat([df, feat_df], axis=1, verify_integrity=True)

    # Removing all the encoded categorical features
    df.drop(cat_cols, axis=1, inplace=True)

    return df
