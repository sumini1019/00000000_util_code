# 2022.05.30 - AUC 만들자!
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sys
from sklearn.metrics import confusion_matrix
from module_sumin.utils_sumin import *


pd.set_option('display.precision', 4)


# df_result_prob = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230220_ELVO_v012_5fold_cross_validation_결과(validation_n1000)\20230222_ELVO 모델 결과 최종(n=1000)\20230222_Result_LVO_Ensemble_Multi_Inference_n1000.csv')
# df_result_prob = read_csv_autodetect_encoding(r'C:\Users\user\Downloads\download_2023-03-15_08-33-05\Result_LVO_Ensemble_Multi_Inference.csv')
df_result_prob = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230316_ELVO_v012_5fold_cross_validation_결과(버그 수정)\Result_LVO_Ensemble_Multi_Inference_all.csv')


df_result_ED = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230220_ELVO_v012_5fold_cross_validation_결과(validation_n1000)\20230221_Result_ED_n1000.csv')
# df_label = read_csv_autodetect_encoding(r'Z:/Stroke/DATA/DATA_cELVO_cASPECTS/Final_List_v2.csv')
df_label = read_csv_autodetect_encoding(r'Z:/Stroke/DATA/DATA_cELVO_cASPECTS/20230222_FinalList_v3_EDO_반구결과포함.csv')


# DataPath 자르기
df_result_prob['DataPath'] = df_result_prob['DataPath'].str.extract("(HR-\w+-\d+)", expand=False)
# 중복 제거
df_result_prob.drop_duplicates(subset=['DataPath'], keep='first', inplace=True)

# Dataframe merge
# - 7개 데이터 빠짐
# - ['HR-A-0374', 'HR-G-0876', 'HR-E-0038', 'HR-A-0438', 'HR-A-0332', 'HR-C-0032', 'HR-M-0009']
merged_df = pd.merge(df_result_prob, df_label, left_on='DataPath', right_on='HR-ID')

# 조건에 따라 열 값 수정
for i, row in merged_df.iterrows():
    if row['LVO-DR (R / L)'] == 'L':
        merged_df.at[i, 'LVO_L'] = 1
        merged_df.at[i, 'LVO_R'] = 0
    elif row['LVO-DR (R / L)'] == 'R':
        merged_df.at[i, 'LVO_L'] = 0
        merged_df.at[i, 'LVO_R'] = 1
    elif row['LVO-DR (R / L)'] == 'N':
        merged_df.at[i, 'LVO_L'] = 0
        merged_df.at[i, 'LVO_R'] = 0
    else:
        raise ValueError('LVO-DR (R / L) column contains invalid values')

# ED 데이터프레임 합치기
merged_df = merged_df.merge(df_result_ED, left_on='DataPath', right_on='Patient_ID')

merged_df = merged_df.drop(['Result_voting_LVO_CHo_LH', 'Result_voting_LVO_CHx_LH', 'Result_voting_EIC_LH', 'Result_voting_DMS_LH', 'Result_voting_OI_LH', 'Result_voting_LVO_CHo_RH', 'Result_voting_LVO_CHx_RH', 'Result_voting_EIC_RH', 'Result_voting_DMS_RH', 'Result_voting_OI_RH', 'Result_mean_LVO_CHo_LH', 'Result_mean_LVO_CHx_LH', 'Result_mean_EIC_LH', 'Result_mean_DMS_LH', 'Result_mean_OI_LH', 'Result_mean_LVO_CHo_RH', 'Result_mean_LVO_CHx_RH', 'Result_mean_EIC_RH', 'Result_mean_DMS_RH', 'Result_mean_OI_RH',
                            '#', 'HR-ID', 'Vender', 'Patient_ID'], axis=1)

merged_df.to_csv('김도현 박사님 전달 버전.csv', index=False)

list_columns = list(merged_df.columns)

merged_df = merged_df.reindex(columns=['DataPath',
 'Prob_LVO_CHo_LH', 'Prob_LVO_CHx_LH', 'Prob_EIC_LH', 'Prob_DMS_LH', 'Prob_OI_LH',
 'Prob_LVO_CHo_RH', 'Prob_LVO_CHx_RH', 'Prob_EIC_RH', 'Prob_DMS_RH', 'Prob_OI_RH',

 'preds_LVO_CHo_fold1_LH', 'preds_LVO_CHo_fold2_LH', 'preds_LVO_CHo_fold3_LH', 'preds_LVO_CHo_fold4_LH', 'preds_LVO_CHo_fold5_LH',
 'preds_LVO_CHx_fold1_LH', 'preds_LVO_CHx_fold2_LH', 'preds_LVO_CHx_fold3_LH', 'preds_LVO_CHx_fold4_LH', 'preds_LVO_CHx_fold5_LH',
 'preds_EIC_fold1_LH', 'preds_EIC_fold2_LH', 'preds_EIC_fold3_LH', 'preds_EIC_fold4_LH', 'preds_EIC_fold5_LH',
 'preds_DMS_fold1_LH', 'preds_DMS_fold2_LH', 'preds_DMS_fold3_LH', 'preds_DMS_fold4_LH', 'preds_DMS_fold5_LH',
 'preds_OI_fold1_LH', 'preds_OI_fold2_LH', 'preds_OI_fold3_LH', 'preds_OI_fold4_LH', 'preds_OI_fold5_LH',

 'preds_LVO_CHo_fold1_RH', 'preds_LVO_CHo_fold2_RH', 'preds_LVO_CHo_fold3_RH', 'preds_LVO_CHo_fold4_RH', 'preds_LVO_CHo_fold5_RH',
 'preds_LVO_CHx_fold1_RH', 'preds_LVO_CHx_fold2_RH', 'preds_LVO_CHx_fold3_RH', 'preds_LVO_CHx_fold4_RH', 'preds_LVO_CHx_fold5_RH',
 'preds_EIC_fold1_RH', 'preds_EIC_fold2_RH', 'preds_EIC_fold3_RH', 'preds_EIC_fold4_RH', 'preds_EIC_fold5_RH',
 'preds_DMS_fold1_RH', 'preds_DMS_fold2_RH', 'preds_DMS_fold3_RH', 'preds_DMS_fold4_RH', 'preds_DMS_fold5_RH',
 'preds_OI_fold1_RH', 'preds_OI_fold2_RH', 'preds_OI_fold3_RH', 'preds_OI_fold4_RH', 'preds_OI_fold5_RH',

 'cELVO_Cls_ED_Left', 'cELVO_Cls_ED_Right', 'cELVO_Prob_ED_Left', 'cELVO_Prob_ED_Right',

 'DMS (Y:1 / N:0)', 'EIC (Y:1 / N:0)', 'GT-LVO (Y:1 / N:0)', 'LVO-DR (R / L)', 'Diag-LVO (P / N)',
                                       'LVO_L', 'LVO_R', 'EIC_L', 'EIC_R', 'DMS_L', 'DMS_R', 'OI_L', 'OI_R'])

def calc_metrics(series_A, series_B, th):
    # 예측값과 정답을 비교하여, True/False 리스트 생성
    prediction = series_A >= th
    ground_truth = series_B.astype(bool)
    correct = prediction == ground_truth

    # 정확도 계산
    accuracy = round(correct.mean(), 4)

    # 민감도 계산
    true_positive = correct & ground_truth
    sensitivity = round((true_positive.sum() / ground_truth.sum()), 4)

    # 특이도 계산
    true_negative = correct & ~ground_truth
    specificity = round((true_negative.sum() / (~ground_truth).sum()), 4)

    # 결과 리턴
    result = pd.Series([accuracy, sensitivity, specificity],
                       index=['Acc', 'Sen', 'Spe'])
    return result

def plot_distribution(df, label, th=0.5):
    for col in df.columns:
        EDO = col[6:9]
        if EDO == 'OI_':
            EDO = 'OI'
        cur_label = label[f'{EDO}_L']

        # 성능 계산
        result_calc = calc_metrics(df[col], cur_label, th)

        plt.hist(df[col], bins=30)
        plt.title(f'{col} / Acc, Sen, Spe : {list(result_calc)}')
        plt.xlabel("Probability")
        plt.ylabel("Number")
        # plt.xlim([-0.5, 1.5])
        plt.show()

def plot_distribution_by_OI(df, label, th=0.5):
    for col in df.columns:
        EDO = col[6:9]
        if EDO == 'OI_':
            EDO = 'OI'
        cur_label = label[f'{EDO}_L']

        # 성능 계산
        result_calc = calc_metrics(df[col], cur_label, th)

        plt.hist(df[col], bins=30)
        plt.title(f'{col} / Acc, Sen, Spe : {list(result_calc)}')
        plt.xlabel("Probability")
        plt.ylabel("Number")
        # plt.xlim([-0.5, 1.5])
        plt.show()


select_df = merged_df[
    ['preds_LVO_CHo_fold1_LH', 'preds_LVO_CHo_fold2_LH', 'preds_LVO_CHo_fold3_LH', 'preds_LVO_CHo_fold4_LH',
     'preds_LVO_CHo_fold5_LH',
     'preds_LVO_CHx_fold1_LH', 'preds_LVO_CHx_fold2_LH', 'preds_LVO_CHx_fold3_LH', 'preds_LVO_CHx_fold4_LH',
     'preds_LVO_CHx_fold5_LH',
     'preds_EIC_fold1_LH', 'preds_EIC_fold2_LH', 'preds_EIC_fold3_LH', 'preds_EIC_fold4_LH', 'preds_EIC_fold5_LH',
     'preds_DMS_fold1_LH', 'preds_DMS_fold2_LH', 'preds_DMS_fold3_LH', 'preds_DMS_fold4_LH', 'preds_DMS_fold5_LH',
     'preds_OI_fold1_LH', 'preds_OI_fold2_LH', 'preds_OI_fold3_LH', 'preds_OI_fold4_LH', 'preds_OI_fold5_LH',

     # 'preds_LVO_CHo_fold1_RH', 'preds_LVO_CHo_fold2_RH', 'preds_LVO_CHo_fold3_RH', 'preds_LVO_CHo_fold4_RH',
     # 'preds_LVO_CHo_fold5_RH',
     # 'preds_LVO_CHx_fold1_RH', 'preds_LVO_CHx_fold2_RH', 'preds_LVO_CHx_fold3_RH', 'preds_LVO_CHx_fold4_RH',
     # 'preds_LVO_CHx_fold5_RH',
     # 'preds_EIC_fold1_RH', 'preds_EIC_fold2_RH', 'preds_EIC_fold3_RH', 'preds_EIC_fold4_RH', 'preds_EIC_fold5_RH',
     # 'preds_DMS_fold1_RH', 'preds_DMS_fold2_RH', 'preds_DMS_fold3_RH', 'preds_DMS_fold4_RH', 'preds_DMS_fold5_RH',
     # 'preds_OI_fold1_RH', 'preds_OI_fold2_RH', 'preds_OI_fold3_RH', 'preds_OI_fold4_RH', 'preds_OI_fold5_RH'
     ]]

label = merged_df[['LVO_L', 'EIC_L', 'DMS_L', 'OI_L']]


# plot_distribution(select_df, label, th=0.5)

plot_distribution_by_OI(select_df, label, th=0.5)