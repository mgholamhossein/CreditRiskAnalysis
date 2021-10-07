import Config
import Preprocessing
import Feature_engineering
import Model
import pandas as pd  # import pandas
import numpy as np

import glob


def read_files():
    file_names = glob.glob(Config.DATA_PATH + "*.csv")
    # read application_train.csv
    train_dataframe = pd.read_csv(file_names[0])
    # read bureau.csv
    bureau_dataframe = pd.read_csv(file_names[1])
    # read bureau_balance.csv
    bureau_balance_dataframe = pd.read_csv(file_names[2])
    # read credit_card_balance.csv
    credit_balance_dataframe = pd.read_csv(file_names[3])
    # read installments_payments.csv
    installments_dataframe = pd.read_csv(file_names[5])
    # read POS_CASH_balance.csv
    POS_CASH_balance_dataframe = pd.read_csv(file_names[6])
    # read previous_application.csv
    prev_app_dataframe = pd.read_csv(file_names[7])
    return [train_dataframe, bureau_dataframe, bureau_balance_dataframe, credit_balance_dataframe,
            installments_dataframe, POS_CASH_balance_dataframe, prev_app_dataframe]


def read_processed_files():
    file_names = glob.glob(Config.DATA_PATH2 + "*Preprocessed.csv")
    dfs = []
    for i in [6, 1, 0, 2, 3, 4, 5]:
        dfs.append(pd.read_csv(file_names[i]))
    return dfs


if __name__ == '__main__':
    # **************** Read the tables *******************
    [train_df, bureau_df, bureau_balance_df, credit_balance_df, installments_df, POS_CASH_balance_df, prev_app_df] = read_files()

    # **************** Preprocessing *******************
    # Remove outliers of the following features - plotting the data distribution is done in the jupiter notebook
    train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    train_df.DAYS_EMPLOYED.fillna(train_df['DAYS_EMPLOYED'].median(), inplace=True)
    train_df["OBS_30_CNT_SOCIAL_CIRCLE"].replace({348: np.nan}, inplace=True)
    train_df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(train_df['OBS_30_CNT_SOCIAL_CIRCLE'].median(), inplace=True)
    train_df["DEF_30_CNT_SOCIAL_CIRCLE"].replace({34: np.nan}, inplace=True)
    train_df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(train_df['DEF_30_CNT_SOCIAL_CIRCLE'].median(), inplace=True)
    train_df["OBS_60_CNT_SOCIAL_CIRCLE"].replace({344: np.nan}, inplace=True)
    train_df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(train_df['OBS_60_CNT_SOCIAL_CIRCLE'].median(), inplace=True)
    train_df["DEF_60_CNT_SOCIAL_CIRCLE"].replace({24: np.nan}, inplace=True)
    train_df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(train_df['DEF_60_CNT_SOCIAL_CIRCLE'].median(), inplace=True)

    train_df_Preprocessed = Preprocessing.preprocessing(train_df, 0)
    train_df_Preprocessed.to_csv(Config.DATA_PATH2 + "train_df_Preprocessed.csv", index=False)

    bureau_df_Preprocessed = Preprocessing.preprocessing(bureau_df, 1)
    bureau_df_Preprocessed.to_csv(Config.DATA_PATH2 + "bureau_df_Preprocessed.csv", index=False)

    bureau_balance_df_Preprocessed = Preprocessing.preprocessing(bureau_balance_df, 2)
    bureau_balance_df_Preprocessed.to_csv(Config.DATA_PATH2 + "bureau_balance_df_Preprocessed.csv", index=False)

    credit_balance_df_Preprocessed = Preprocessing.preprocessing(credit_balance_df, 3)
    credit_balance_df_Preprocessed.to_csv(Config.DATA_PATH2 + "credit_balance_df_Preprocessed.csv", index=False)

    installments_df_Preprocessed = Preprocessing.preprocessing(installments_df, 5)
    installments_df_Preprocessed.to_csv(Config.DATA_PATH2 + "installments_df_Preprocessed.csv", index=False)

    POS_CASH_balance_df_Preprocessed = Preprocessing.preprocessing(POS_CASH_balance_df, 6)
    POS_CASH_balance_df_Preprocessed.to_csv(Config.DATA_PATH2 + "POS_CASH_balance_df_Preprocessed.csv",
                                            index=False)

    prev_app_df_Preprocessed = Preprocessing.preprocessing(prev_app_df, 7)
    prev_app_df_Preprocessed.to_csv(Config.DATA_PATH2 + "prev_app_df_Preprocessed.csv", index=False)


    # **************** Feature Engineering *******************
    processed_dataframes_list = [main_df, bureau_df, bureau_balance_df, credit_balance_df, installments_df,
                                 POS_CASH_balance_df, prev_app_df] = read_processed_files()
    for i in processed_dataframes_list:
        # print(f'i shape before {i.shape}')
        i = Feature_engineering.drop_high_corr_data(i)
        # print(f'i shape after {i.shape}')

    main_df_before_agg = main_df
    main_df_before_agg.to_csv(Config.DATA_PATH2 + "before_aggregated_df.csv", index=False)
    aggregated_df = Feature_engineering.agg_with_main_df(processed_dataframes_list)
    print('aggregated_df', aggregated_df.shape)
    print((aggregated_df.isnull().sum().sum()))
    aggregated_df = Feature_engineering.drop_high_corr_data(aggregated_df)
    print('aggregated_df after dropping high correlated features', aggregated_df.shape)
    aggregated_df.to_csv(Config.DATA_PATH2 + "aggregated_df.csv", index=False)

    # **************** Model *******************
    final_models = Model.run_model()
    Model.check_model_importance()

