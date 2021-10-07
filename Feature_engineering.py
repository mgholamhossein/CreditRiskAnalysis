import numpy as np  # import numpy
import Config
import copy

def drop_high_corr_data(df):
    print(type(df))
    fts = [f for f in df.columns if f not in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', 'TARGET']]
    fts2 = [f for f in df.columns if f in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', 'TARGET']]
    corr_mat = df[fts].corr()
    print(f'corr_mat.shape = {corr_mat.shape}')
    feat_to_remove = []
    lo_drop = []

    while len(lo_drop) != 0:
        corr_matrix = corr_mat.abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than threshold (0.9)
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

        # Select lower triangle of correlation matrix
        lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(np.bool))
        # Find features with correlation greater than threshold (0.9)
        lo_drop = [column for column in lower.columns if any(lower[column] > 0.9)]

        # feat_to_remove = []
        for i in range(len(lo_drop)):
            if df[lo_drop[i]].var() > df[to_drop[i]].var():
                feat_to_remove.append(to_drop[i])
            else:
                feat_to_remove.append(lo_drop[i])

        fts1 = [f for f in df.columns if f not in feat_to_remove + fts2]
        corr_mat = df[fts1].corr()
        print(f'corr_mat.shape = {corr_mat.shape}')

    print(f'len(feat_to_remove) = {len(feat_to_remove)}')

    df = df.drop(feat_to_remove, 1)
    print(df.shape)
    return df


def fill_missing_values(dff, miss_cols):
    # missing_cols = [f for f in df_list[0].columns if df_list[0][f].isnull().sum() > 0]
    cat_cols = [f for f in miss_cols if 'object' in str(dff[f].dtype)]
    num_cols = [f for f in miss_cols if 'object' not in str(dff[f].dtype)]

    if len(cat_cols) != 0:
        for cat_feat in cat_cols:
            dff[cat_feat].fillna('unknown', inplace=True)
    if len(num_cols) != 0:
        for num_feat in num_cols:
            dff[num_feat].fillna(0, inplace=True)  # dff[num_feat].median()

    print(dff.shape)
    print(dff.isnull().sum().sum())
    return dff


def agg_with_main_df(df_list):
    tot_agg_dict = {
        'EXT_SOURCE_1': ['min', 'max', 'sum', 'mean', 'median'],
        'EXT_SOURCE_2': ['min', 'max', 'sum', 'mean', 'median'],
        'EXT_SOURCE_3': ['min', 'max', 'sum', 'mean', 'median'],
        'ENDDATE_DIF': ['sum', 'mean', 'max', 'min', 'median'],
        'DEBT_PERCENT': ['sum', 'mean', 'max', 'min', 'median'],
        'DEBT_CREDIT_DIFF': ['sum', 'mean', 'max', 'min', 'median'],
        'dept_limit_rate': ['sum', 'mean', 'max', 'min', 'median'],
        'credit_update_days_diff': ['sum', 'mean', 'max', 'min', 'median'],
        'bureau_credit_active_binary': ['nunique'],
        'bureau_credit_enddate_binary': ['nunique'],
        'ACTUAL_DURATION_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'DIFF_DURATION_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'SUPPOSED_DURATION_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_ANNUITY': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_DOWN_PAYMENT': ['sum', 'mean', 'max', 'min', 'median'],
        'NFLAG_LAST_APPL_IN_DAY': ['nunique'],
        'NFLAG_INSURED_ON_APPROVAL': ['nunique'],
        'PRODUCT_COMBINATION_enc': ['nunique'],
        'CODE_REJECT_REASON_enc': ['nunique'],
        'NAME_CASH_LOAN_PURPOSE_enc': ['nunique'],
        'AMT_PAYMENT': ['sum', 'mean', 'max', 'min', 'median'],
        'NUM_INSTALMENT_NUMBER': ['sum', 'mean', 'max', 'min', 'median'],
        'MONTHS_BALANCE': ['sum', 'mean', 'max', 'min', 'median'],
        # 'MONTHS_BALANCE_bb_count': ['sum', 'mean', 'max', 'min', 'median'],
        # 'MONTHS_BALANCE_pos_count': ['sum', 'mean', 'max', 'min', 'median'],
        # 'MONTHS_BALANCE_cb_count': ['sum', 'mean', 'max', 'min', 'median'],
        # 'MONTHS_BALANCE_CB': ['sum', 'mean', 'max', 'min', 'median'],
        # 'MONTHS_BALANCE_POS': ['sum', 'mean', 'max', 'min', 'median'],
        'STATUS_enc': ['nunique'],
        'credit_bureau_status': ['nunique'],
        'amt_payment_total_balance_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'amt_payment_balance_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'balance_credit_limit_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'total_cnt_drawing_instalment_num_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'DAYS_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'CREDIT_DAY_OVERDUE': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_CREDIT_MAX_OVERDUE': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max', 'min', 'median'],
        'DAYS_CREDIT_UPDATE': ['sum', 'mean', 'max', 'min', 'median'],
        'CREDIT_ACTIVE_enc': ['nunique'],
        'CREDIT_CURRENCY_enc': ['nunique'],
        'CREDIT_TYPE_enc': ['nunique'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_DRAWINGS_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_DRAWINGS_POS_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'AMT_INST_MIN_REGULARITY': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_DRAWINGS_ATM_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_DRAWINGS_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_INSTALMENT_MATURE_CUM': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_INSTALMENT': ['sum', 'mean', 'max', 'min', 'median'],
        'CNT_INSTALMENT_FUTURE': ['sum', 'mean', 'max', 'min', 'median'],
        'SK_DPD': ['sum', 'mean', 'max', 'min', 'count', 'median'],
        'SK_DPD_DEF': ['sum', 'mean', 'max', 'min', 'count', 'median'],
        'NAME_CONTRACT_STATUS_enc': ['nunique'],
        'instalment_paid_late_in_days': ['sum', 'mean', 'max', 'min', 'median'],
        'amt_payment_installment_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'instalment_paid_late': ['sum', 'mean', 'max', 'min', 'count', 'median'],
        'instalment_paid_over_amount': ['sum', 'mean', 'max', 'min', 'median'],
        'instalment_paid_over': ['sum', 'mean', 'max', 'min', 'count', 'median'],
        'INSTALMENT_FUTURE_installment_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'RATE_AMT_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'RATE_ANN_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'RATE_DOWNPAY_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'RATE_GOODS_CREDIT': ['sum', 'mean', 'max', 'min', 'median'],
        'RATE_ANN_APP': ['sum', 'mean', 'max', 'min', 'median'],
        'prev_amt_application_credit_ratio': ['sum', 'mean', 'max', 'min', 'median'],
        'SK_ID_BUREAU_count': ['count'],
        'bureau_annuity_credit_ratio': ['sum', 'mean', 'max', 'min', 'median']
    }

    # [ main_df, bureau_df, bureau_balance_df, credit_balance_df, installments_df, POS_CASH_balance_df, prev_app_df]
    new_list = [df_list[2], df_list[1], df_list[3], df_list[4], df_list[5], df_list[6]]

    # Engineered features for main_df : df_list[0]
    df_list[0]['kids_family_ratio'] = df_list[0]['CNT_CHILDREN'] / df_list[0]['CNT_FAM_MEMBERS']
    df_list[0]['income_per_fam_member'] = df_list[0]['AMT_INCOME_TOTAL'] / df_list[0]['CNT_FAM_MEMBERS']
    df_list[0]['income_per_child'] = df_list[0]['AMT_INCOME_TOTAL'] / df_list[0]['CNT_CHILDREN']
    df_list[0]['employment_age_ratio'] = df_list[0]['DAYS_EMPLOYED'] / df_list[0]['DAYS_BIRTH']
    df_list[0]['annuity_income_ratio'] = df_list[0]['AMT_ANNUITY'] / df_list[0]['AMT_INCOME_TOTAL']
    df_list[0]['annuity_credit_ratio'] = df_list[0]['AMT_ANNUITY']/df_list[0]['AMT_CREDIT']
    df_list[0]['annuity_goods_ratio'] = df_list[0]['AMT_ANNUITY']/df_list[0]['AMT_GOODS_PRICE']
    df_list[0]['credit_goods_ratio'] = df_list[0]['AMT_CREDIT']/df_list[0]['AMT_GOODS_PRICE']
    df_list[0]['credit_income_ratio'] = df_list[0]['AMT_CREDIT']/df_list[0]['AMT_INCOME_TOTAL']
    df_list[0]['income_employed_ratio'] = df_list[0]['AMT_INCOME_TOTAL']/df_list[0]['DAYS_EMPLOYED']
    df_list[0]['income_birth_ratio'] = df_list[0]['AMT_INCOME_TOTAL']/df_list[0]['DAYS_BIRTH']
    df_list[0]['ID_birth_ratio'] = df_list[0]['DAYS_ID_PUBLISH']/df_list[0]['DAYS_BIRTH']
    df_list[0]['car_birth_ratio'] = df_list[0]['OWN_CAR_AGE']/df_list[0]['DAYS_BIRTH']
    df_list[0]['car_employed_ratio'] = df_list[0]['OWN_CAR_AGE']/df_list[0]['DAYS_EMPLOYED']
    df_list[0]['annuity_employed_ratio'] = df_list[0]['AMT_ANNUITY'] / df_list[0]['DAYS_EMPLOYED']

    # External sources
    df_list[0]['Ext_production'] = df_list[0]['EXT_SOURCE_1'] * df_list[0]['EXT_SOURCE_2'] * df_list[0]['EXT_SOURCE_3']
    df_list[0]['EXT_SRC_weighted3'] = (df_list[0]['EXT_SOURCE_1'] * 2 + df_list[0]['EXT_SOURCE_2'] * 3 + df_list[0]['EXT_SOURCE_3'] * 4) / 9
    df_list[0]['EXT_SRC_weighted2'] = (df_list[0]['EXT_SOURCE_1'] * 3 + df_list[0]['EXT_SOURCE_2'] * 4 + df_list[0]['EXT_SOURCE_3'] * 2) / 9
    df_list[0]['EXT_SRC_weighted1'] = (df_list[0]['EXT_SOURCE_1'] * 4 + df_list[0]['EXT_SOURCE_2'] * 2 + df_list[0]['EXT_SOURCE_3'] * 3) / 9

    # for function_name in ['min', 'max', 'sum', 'mean', 'median']:
    #     df_list[0]['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
    #         df_list[0][['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    df_list[0]['registration-ID-days-diff'] = df_list[0]['DAYS_REGISTRATION'] - df_list[0]['DAYS_ID_PUBLISH']
    df_list[0]['phone_to_birth_ratio'] = df_list[0]['DAYS_LAST_PHONE_CHANGE'] / df_list[0]['DAYS_BIRTH']
    df_list[0]['phone_to_employ_ratio'] = df_list[0]['DAYS_LAST_PHONE_CHANGE'] / df_list[0]['DAYS_EMPLOYED']

    df_list[0]['payment_rate'] = df_list[0]['AMT_ANNUITY'] / df_list[0]['AMT_CREDIT']

    # Engineered features for bureau_df : df_list[1]
    # supposed credit duration:
    df_list[1]['SUPPOSED_DURATION_CREDIT'] = df_list[1]['DAYS_CREDIT_ENDDATE'] + df_list[1]['DAYS_CREDIT']
    df_list[1]['ENDDATE_DIF'] = df_list[1]['DAYS_CREDIT_ENDDATE']- df_list[1]['DAYS_ENDDATE_FACT']
    df_list[1]['DEBT_PERCENT'] =  df_list[1]['AMT_CREDIT_SUM']/ df_list[1]['AMT_CREDIT_SUM_DEBT']
    df_list[1]['DEBT_CREDIT_DIFF'] = df_list[1]['AMT_CREDIT_SUM'] - df_list[1]['AMT_CREDIT_SUM_DEBT']
    df_list[1]['dept_limit_rate'] = df_list[1]['AMT_CREDIT_SUM_DEBT'] / df_list[1]['AMT_CREDIT_SUM_LIMIT']
    df_list[1]['credit_update_days_diff'] = df_list[1]['DAYS_CREDIT_UPDATE'] - df_list[1]['DAYS_CREDIT']

    df_list[1]['bureau_credit_active_binary'] = (df_list[1]['CREDIT_ACTIVE_enc'] != 2).astype(int)
    df_list[1]['bureau_credit_enddate_binary'] = (df_list[1]['DAYS_CREDIT_ENDDATE'] > 0).astype(int)

    # actual credit duration:
    df_list[1]['ACTUAL_DURATION_CREDIT'] = df_list[1]['DAYS_ENDDATE_FACT'] - df_list[1]['DAYS_CREDIT']
    # diff between credit duration and actual duration:
    df_list[1]['DIFF_DURATION_CREDIT'] = df_list[1]['ACTUAL_DURATION_CREDIT'] - df_list[1]['SUPPOSED_DURATION_CREDIT']

    df1 = df_list[1].groupby('SK_ID_CURR').agg({'SK_ID_BUREAU': 'count'})
    # modify the aggregated column name
    df1.columns = ['SK_ID_BUREAU_count']
    df_list[0] = df_list[0].merge(df1, how='left', on='SK_ID_CURR')
    df_list[0]['SK_ID_BUREAU_count'] = df_list[0]['SK_ID_BUREAU_count'].fillna(0)

    df_list[1]['bureau_annuity_credit_ratio'] = df_list[1]['AMT_ANNUITY'] / df_list[1]['AMT_CREDIT_SUM_LIMIT']
    df_list[1].to_csv(Config.DATA_PATH2 + "df_list1_tp1.csv", index=False)
    # Engineered features for bureau_balance_df : df_list[2]
    df_list[2]['credit_bureau_status'] = df_list[2]['STATUS_enc'].apply(lambda x: 0 if x in [6, 0, 3] else 1)

    df_list[2].sort_values('MONTHS_BALANCE', inplace=True, ascending=False)
    df_list[2]['MONTHS_BALANCE_bb_count'] = df_list[2].groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].count()

    # Engineered features for credit_balance_df : df_list[3]
    # df_list[3].rename(columns={'MONTHS_BALANCE': 'MONTHS_BALANCE_CB'}, inplace=True)
    df_list[3]['amt_payment_total_balance_ratio'] = df_list[3]['AMT_PAYMENT_TOTAL_CURRENT']/df_list[3]['AMT_BALANCE']
    df_list[3]['amt_payment_balance_ratio'] = df_list[3]['AMT_PAYMENT_CURRENT'] / df_list[3]['AMT_BALANCE']
    df_list[3]['balance_credit_limit_ratio'] = df_list[3]['AMT_BALANCE'] / df_list[3]['AMT_CREDIT_LIMIT_ACTUAL']
    df_list[3]['total_cnt_drawing_instalment_num_ratio'] = (df_list[3]['CNT_DRAWINGS_ATM_CURRENT'] +
                                                            df_list[3]['CNT_DRAWINGS_CURRENT'] +
                                                            df_list[3]['CNT_DRAWINGS_OTHER_CURRENT'] +
                                                            df_list[3]['CNT_DRAWINGS_POS_CURRENT'])\
                                                           / df_list[3]['CNT_INSTALMENT_MATURE_CUM']

    df_list[3].sort_values('MONTHS_BALANCE', inplace=True, ascending=False)
    df_list[3]['MONTHS_BALANCE_cb_count'] = df_list[3].groupby('SK_ID_CURR')['MONTHS_BALANCE'].count()

    # get the highest value of each group
    df_test1 = copy.copy(df_list[3].groupby('SK_ID_CURR').head(1))
    df_test1.reset_index(drop=True)
    df_test1['amt_credit_limit_balance_diff'] = df_test1['AMT_CREDIT_LIMIT_ACTUAL'] - df_test1['AMT_BALANCE']
    df_test1['balance_amt_credit_limit_ratio'] = df_test1['AMT_BALANCE'] / df_test1['AMT_CREDIT_LIMIT_ACTUAL']

    df_final1 = df_test1.groupby('SK_ID_CURR').agg({'amt_credit_limit_balance_diff': ['max', 'min', 'mean'],
                                                    'balance_amt_credit_limit_ratio': ['max', 'min', 'mean']})
    # modify the aggregated column name
    df_final1.columns = ['{}_{}'.format(f[0], f[1]) for f in df_final1.columns]
    df_list[0] = df_list[0].merge(df_final1, how='left', on='SK_ID_CURR')

    # Engineered features for installments_df : df_list[4]
    df_list[4]['instalment_paid_late_in_days'] = df_list[4]['DAYS_ENTRY_PAYMENT']-df_list[4]['DAYS_INSTALMENT']
    df_list[4]['amt_payment_installment_ratio'] = df_list[4]['AMT_PAYMENT']/df_list[4]['AMT_INSTALMENT']

    df_list[4]['instalment_paid_late'] = (df_list[4]['instalment_paid_late_in_days'] > 0).astype(int)
    df_list[4]['instalment_paid_over_amount'] = df_list[4]['AMT_PAYMENT'] - df_list[4]['AMT_INSTALMENT']
    df_list[4]['instalment_paid_over'] = (df_list[4]['instalment_paid_over_amount'] > 0).astype(int)


    # df_list[4]['amt_payment_installment_diff'] = df_list[4]['Entry-Installment-diff'] / df_list[4]['amt_payment_installment_ratio']

    # group by current ID and then sort DAYS_INSTALLMENT of each group
    df = df_list[4].groupby(['SK_ID_CURR'], sort=False).apply(
        lambda x: x.sort_values(['DAYS_INSTALMENT'], ascending=True)).reset_index(drop=True)
    # get the highest 5 values of each group
    df_test = df.groupby('SK_ID_CURR').head(5)
    # take the mean of these highest five values for each group
    df_final = df_test.groupby('SK_ID_CURR').agg({'DAYS_INSTALMENT': ['mean', 'min', 'max', 'median', 'sum', 'count']})
    # modify the aggregated column name
    df_final.columns = ['DAYS_INSTALMENT_mean', 'DAYS_INSTALMENT_min', 'DAYS_INSTALMENT_max', 'DAYS_INSTALMENT_median',
                        'DAYS_INSTALMENT_sum', 'DAYS_INSTALMENT_count']
    df_list[0] = df_list[0].merge(df_final, how='left', on='SK_ID_CURR')

    # Engineered features for POS_CASH_balance_df : df_list[5]
    # df_list[5].rename(columns={'MONTHS_BALANCE': 'MONTHS_BALANCE_POS'}, inplace=True)
    df_list[5]['INSTALMENT_FUTURE_installment_ratio'] = df_list[5]['CNT_INSTALMENT_FUTURE'] / df_list[5]['CNT_INSTALMENT']
    df_list[5].sort_values('MONTHS_BALANCE', inplace=True, ascending=False)
    df_list[5]['MONTHS_BALANCE_pos_count'] = df_list[5].groupby('SK_ID_CURR')['MONTHS_BALANCE'].count()

    # Engineered features for prev_app_df : df_list[6]
    df_list[6]['RATE_AMT_CREDIT'] = df_list[6]['AMT_APPLICATION'] / df_list[6]['AMT_CREDIT']
    df_list[6]['RATE_ANN_CREDIT'] = df_list[6]['AMT_ANNUITY'] / df_list[6]['AMT_CREDIT']
    df_list[6]['RATE_DOWNPAY_CREDIT'] = df_list[6]['AMT_DOWN_PAYMENT'] / df_list[6]['AMT_CREDIT']
    df_list[6]['RATE_DOWNPAY_CREDIT'] = df_list[6]['RATE_DOWNPAY_CREDIT'].apply(lambda x: np.clip(x, 0, 1))
    df_list[6]['RATE_GOODS_CREDIT'] = df_list[6]['AMT_GOODS_PRICE'] / df_list[6]['AMT_CREDIT']
    df_list[6]['RATE_ANN_APP'] = df_list[6]['AMT_ANNUITY'] / df_list[6]['AMT_APPLICATION']
    df_list[6]['prev_amt_application_credit_ratio'] = df_list[6]['AMT_APPLICATION'] / df_list[6]['AMT_CREDIT']

    df_list[6]['prev_app_refused'] = (df_list[6]['NAME_CONTRACT_STATUS_enc'] == 2).astype(int)
    prev_group = df_list[6].groupby('SK_ID_CURR').agg({'SK_ID_PREV': ['count'], 'prev_app_refused': ['count']})
    prev_group.columns = ['Prev_ID_count', 'prev_refused_app_count']
    df_list[0] = df_list[0].merge(prev_group, how='left', on='SK_ID_CURR')

    # RATE_INTEREST_PRIMARY

    # df_list[6].drop(
    #     ['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_DOWN_PAYMENT', 'RATE_DOWNPAY_CREDIT', 'AMT_GOODS_PRICE'],
    #     axis=1, inplace=True)

    # # [ main_df, bureau_df, bureau_balance_df, credit_balance_df, installments_df, POS_CASH_balance_df, prev_app_df]
    # # new_list = [df_list[2], df_list[1], df_list[3], df_list[4], df_list[5], df_list[6]]

    # always groupby SK_ID_CURR because this is the key to link with the main table
    for df in new_list:

        temp_list = [f for f in tot_agg_dict.keys() if f in df.columns]
        agg_dict = {x: tot_agg_dict[x] for x in temp_list}
        if df is df_list[2]:
            print('agg_dict df_list[2]', agg_dict)
            temp_df = df_list[2].groupby('SK_ID_BUREAU').agg(agg_dict)
            temp_df.columns = ['{}_{}'.format(f[0], f[1]) for f in temp_df.columns]
            # temp_df.head()

            # print(df_list[1].shape)
            df_list[1] = df_list[1].merge(temp_df, how='left', on='SK_ID_BUREAU')
            # print(df_list[1].shape)

            # we shall fill them again
            missing_cols = [f for f in df_list[1].columns if df_list[1][f].isnull().sum() > 0]
            df_list[1] = fill_missing_values(df_list[1], missing_cols)

        else:
            if df is df_list[1]:
                print('agg_dict df_list[1]', agg_dict)
            temp_df = df.groupby('SK_ID_CURR').agg(agg_dict)
            temp_df.columns = ['{}_{}'.format(f[0], f[1]) for f in temp_df.columns]
            # temp_df.head()

            # print(df_list[0].shape)
            df_list[0] = df_list[0].merge(temp_df, how='left', on='SK_ID_CURR')
            # print(df_list[0].shape)

            # we shall fill them again
            missing_cols = [f for f in df_list[0].columns if df_list[0][f].isnull().sum() > 0]
            print('df  ', len(missing_cols))

            df_list[0] = fill_missing_values(df_list[0], missing_cols)

    return df_list[0]
