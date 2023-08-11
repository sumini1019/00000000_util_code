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
from sklearn.metrics import roc_curve, auc


pd.set_option('display.precision', 4)


# df_result_prob = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230220_ELVO_v012_5fold_cross_validation_결과(validation_n1000)\20230222_ELVO 모델 결과 최종(n=1000)\20230222_Result_LVO_Ensemble_Multi_Inference_n1000.csv')
# df_result_prob = read_csv_autodetect_encoding(r'C:\Users\user\Downloads\download_2023-03-15_08-33-05\Result_LVO_Ensemble_Multi_Inference.csv')
# df_result_prob = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230316_ELVO_v012_5fold_cross_validation_결과(버그 수정)\Result_LVO_Ensemble_Multi_Inference_all.csv')

# 2023.04.19
# - New Model (Slice Number 오류 수정)
# df_result_prob = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230419_ELVO_재학습모델_cross_validation_결과(validation_n1000)\Result_LVO_Ensemble_Multi_Inference_modelA.csv')
df_result_prob = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230419_ELVO_재학습모델_cross_validation_결과(validation_n1000)\Result_LVO_Ensemble_Multi_Inference_modelB.csv')
df_result_ED = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230419_ELVO_재학습모델_cross_validation_결과(validation_n1000)\20230221_Result_ED_n1000.csv')
df_label = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230419_ELVO_재학습모델_cross_validation_결과(validation_n1000)\20230222_FinalList_v3_EDO_반구결과포함.csv')

# DataPath 자르기
df_result_prob['DataPath'] = df_result_prob['DataPath'].str.extract("(HR-\w+-\d+)", expand=False)

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

# ED column 분할 (LH)
new_cols = ['LH_Prob_ED_Normal', 'LH_Prob_ED_R', 'LH_Prob_ED_L']
merged_df[new_cols] = merged_df['cELVO_Prob_ED_Left'].apply(lambda x: pd.Series([round(float(val.strip()), 4) for val in x.strip('[]').split()]))
merged_df.drop('cELVO_Prob_ED_Left', axis=1, inplace=True)
# ED column 분할 (RH)
new_cols = ['RH_Prob_ED_Normal', 'RH_Prob_ED_R', 'RH_Prob_ED_L']
merged_df[new_cols] = merged_df['cELVO_Prob_ED_Right'].apply(lambda x: pd.Series([round(float(val.strip()), 4) for val in x.strip('[]').split()]))
merged_df.drop('cELVO_Prob_ED_Right', axis=1, inplace=True)

# Merged DF 컬럼 순서 변경
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

 'cELVO_Cls_ED_Left', 'cELVO_Cls_ED_Right',
 'LH_Prob_ED_Normal', 'LH_Prob_ED_R', 'LH_Prob_ED_L', 'RH_Prob_ED_Normal', 'RH_Prob_ED_R', 'RH_Prob_ED_L',
 # 'cELVO_Cls_ED_Left', 'cELVO_Cls_ED_Right', 'cELVO_Prob_ED_Left', 'cELVO_Prob_ED_Right',

 'DMS (Y:1 / N:0)', 'EIC (Y:1 / N:0)', 'GT-LVO (Y:1 / N:0)', 'LVO-DR (R / L)', 'Diag-LVO (P / N)',
                                       'LVO_L', 'LVO_R', 'EIC_L', 'EIC_R', 'DMS_L', 'DMS_R', 'OI_L', 'OI_R'])

merged_df.to_csv('김도현 박사님 전달 버전.csv', index=False)

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
        if 'CHx' in col:
            continue

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


plot_distribution(select_df, label, th=0.5)


def eval_ELVO_performance(merged_df, branch_by_OI=False, threshold_OI=0.5, threshold_LVO=0.5, draw_ROC=False):
    # Setting
    hemi_tp = hemi_fp = hemi_tn = hemi_fn = 0
    series_tp = series_fp = series_tn = series_fn = 0
    list_hemi = ['LH', 'RH']
    total_count = len(merged_df)

    hemi_probs = []
    hemi_labels = []
    series_probs = []
    series_labels = []

    for index, row in merged_df.iterrows():
        # 환자 기준 결과,Label의 양/음성 변수 초기화
        series_Result = False
        series_Label = False
        series_Prob = -1

        # 방향 별 확인
        for direction in list_hemi:
            # 현재 방향의 prob (OI 분기)
            if branch_by_OI:
                prob_chx = row[f'Prob_LVO_CHx_{direction}']
                prob_cho = row[f'Prob_LVO_CHo_{direction}']
                # - OI 여부에 따라, 반구비교/반구비교제거 모델 prob 사용
                hemi_prob = prob_chx if (row['Prob_OI_RH'] > threshold_OI) or (row['Prob_OI_LH'] > threshold_OI) else prob_cho

            else:
                hemi_prob = row[f'Prob_LVO_CHo_{direction}']
            # 현재 방향의 Label
            hemi_label = row[f'LVO_{direction[0]}']

            # 반구 기준 - Mertric 계산
            if hemi_prob >= threshold_LVO and hemi_label == 1:
                hemi_tp += 1
            elif hemi_prob < threshold_LVO and hemi_label == 0:
                hemi_tn += 1
            elif hemi_prob >= threshold_LVO and hemi_label == 0:
                hemi_fp += 1
            else:
                hemi_fn += 1

            # 환자 기준 - 결과/Label 체크
            if hemi_prob >= threshold_LVO:
                series_Result = True
            if hemi_label == 1:
                series_Label = True
            if hemi_prob > series_Prob:
                series_Prob = hemi_prob

            # Hemi AUC 계산 위해서, Prob, Label 값을 리스트에 추가
            hemi_probs.append(hemi_prob)
            hemi_labels.append(hemi_label)

        # Series AUC 계산 위해서, Prob, Label 값을 리스트에 추가
        series_probs.append(series_Prob)
        series_labels.append(series_Label)

        # 환자 기준 - Metric 계산
        if series_Result and series_Label:
            series_tp += 1
        elif not series_Result and not series_Label:
            series_tn += 1
        elif series_Result and not series_Label:
            series_fp += 1
        else:
            series_fn += 1

    # Series / Hemi 최종 Metric 계산
    series_ACC = (series_tp + series_tn) / (series_tp + series_tn + series_fp + series_fn)
    series_SEN = series_tp / (series_tp + series_fn)
    series_SPE = series_tn / (series_tn + series_fp)
    series_AUC = plot_roc_curve(series_labels, series_probs, 'LVO (Series)', draw_ROC)  # ROC Curve, AUC 계산

    hemi_ACC = (hemi_tp + hemi_tn) / (hemi_tp + hemi_tn + hemi_fp + hemi_fn)
    hemi_SEN = hemi_tp / (hemi_tp + hemi_fn)
    hemi_SPE = hemi_tn / (hemi_tn + hemi_fp)
    hemi_AUC = plot_roc_curve(hemi_labels, hemi_probs, 'LVO (Hemi)', draw_ROC)  # ROC Curve, AUC 계산

    # 결과 묶기
    result_Series = {"acc": series_ACC,
                     "sen": series_SEN,
                     "spe": series_SPE,
                     "auc": series_AUC,
                     "tp": series_tp,
                     "fp": series_fp,
                     "tn": series_tn,
                     "fn": series_fn,
                     "total_count": total_count
                     }
    result_Hemi = {"acc": hemi_ACC,
                   "sen": hemi_SEN,
                   "spe": hemi_SPE,
                   "auc": hemi_AUC,
                   "tp": hemi_tp,
                   "fp": hemi_fp,
                   "tn": hemi_tn,
                   "fn": hemi_fn,
                   "total_count": total_count
                   }


    return result_Series, result_Hemi

# 각 성능 지표를 리스트에 저장합니다.
Series_acc_list = []
Series_sen_list = []
Series_spe_list = []
Series_auc_list = []
Hemi_acc_list = []
Hemi_sen_list = []
Hemi_spe_list = []
Hemi_auc_list = []


result_Series, result_Hemi = eval_ELVO_performance(merged_df, branch_by_OI=True, draw_ROC=True)
Series_acc_list.append(result_Series["acc"])
Series_sen_list.append(result_Series["sen"])
Series_spe_list.append(result_Series["spe"])
Series_auc_list.append(result_Series["auc"])
Hemi_acc_list.append(result_Hemi["acc"])
Hemi_sen_list.append(result_Hemi["sen"])
Hemi_spe_list.append(result_Hemi["spe"])
Hemi_auc_list.append(result_Hemi["auc"])

print('\n### 1. Branch by OI')
print('1-1. Series 기준')
print(f'Accuracy : {result_Series["acc"]:.4f}')
print(f'Sensitivity : {result_Series["sen"]:.4f}')
print(f'Specificity : {result_Series["spe"]:.4f}')
print(f'AUC : {result_Series["auc"]:.4f}')
print(f'TP : {result_Series["tp"]}, FP : {result_Series["fp"]}, TN : {result_Series["tn"]}, FN : {result_Series["fn"]}, '
      f'Total : {result_Series["total_count"]}')
print('1-2. Hemi 기준')
print(f'Accuracy : {result_Hemi["acc"]:.4f}')
print(f'Sensitivity : {result_Hemi["sen"]:.4f}')
print(f'Specificity : {result_Hemi["spe"]:.4f}')
print(f'AUC : {result_Hemi["auc"]:.4f}')
print(f'TP : {result_Hemi["tp"]}, FP : {result_Hemi["fp"]}, TN : {result_Hemi["tn"]}, FN : {result_Hemi["fn"]}, '
      f'Total : {result_Hemi["total_count"]}')

result_Series, result_Hemi = eval_ELVO_performance(merged_df, branch_by_OI=False, draw_ROC=True)
Series_acc_list.append(result_Series["acc"])
Series_sen_list.append(result_Series["sen"])
Series_spe_list.append(result_Series["spe"])
Series_auc_list.append(result_Series["auc"])
Hemi_acc_list.append(result_Hemi["acc"])
Hemi_sen_list.append(result_Hemi["sen"])
Hemi_spe_list.append(result_Hemi["spe"])
Hemi_auc_list.append(result_Hemi["auc"])

print('\n### 2. Do not use OI')
print('2-1. Series 기준')
print(f'Accuracy : {result_Series["acc"]:.4f}')
print(f'Sensitivity : {result_Series["sen"]:.4f}')
print(f'Specificity : {result_Series["spe"]:.4f}')
print(f'AUC : {result_Series["auc"]:.4f}')
print(f'TP : {result_Series["tp"]}, FP : {result_Series["fp"]}, TN : {result_Series["tn"]}, FN : {result_Series["fn"]}, '
      f'Total : {result_Series["total_count"]}')
print('2-2. Hemi 기준')
print(f'Accuracy : {result_Hemi["acc"]:.4f}')
print(f'Sensitivity : {result_Hemi["sen"]:.4f}')
print(f'Specificity : {result_Hemi["spe"]:.4f}')
print(f'AUC : {result_Hemi["auc"]:.4f}')
print(f'TP : {result_Hemi["tp"]}, FP : {result_Hemi["fp"]}, TN : {result_Hemi["tn"]}, FN : {result_Hemi["fn"]}, '
      f'Total : {result_Hemi["total_count"]}')



def find_optimal_thresholds(merged_df, branch_by_OI=True, optimal_by_Series=True):
    best_result_Series = {"acc": 0,
                     "sen": 0,
                     "spe": 0,
                     "tp": 0,
                     "fp": 0,
                     "tn": 0,
                     "fn": 0,
                     "best_thresh_oi": None,
                     "best_thresh_lvo": None
                     }
    best_result_Hemi = {"acc": 0,
                   "sen": 0,
                   "spe": 0,
                   "tp": 0,
                   "fp": 0,
                   "tn": 0,
                   "fn": 0,
                   "best_thresh_oi": None,
                   "best_thresh_lvo": None
                   }

    for thresh_oi in np.arange(0.1, 1.0, 0.1):
        for thresh_lvo in np.arange(0.1, 1.0, 0.1):
            result_Series, result_Hemi = eval_ELVO_performance(merged_df, branch_by_OI=branch_by_OI,
                                                               threshold_OI=thresh_oi, threshold_LVO=thresh_lvo,
                                                               draw_ROC=False)

            # Optimize의 기준 선택 (Series or Hemi)
            if optimal_by_Series:
                # 최적값 업데이트
                avg_metric = (result_Series["acc"] + result_Series["sen"] + result_Series["spe"]) / 3
            else:
                avg_metric = (result_Hemi["acc"] + result_Hemi["sen"] + result_Hemi["spe"]) / 3
            # 기존 Best와 비교 후 업데이트
            if avg_metric > (best_result_Series["acc"] + best_result_Series["sen"] + best_result_Series["spe"]) / 3:
                best_result_Series["acc"] = result_Series["acc"]
                best_result_Series["sen"] = result_Series["sen"]
                best_result_Series["spe"] = result_Series["spe"]
                best_result_Series["auc"] = result_Series["auc"]
                best_result_Series["tp"] = result_Series["tp"]
                best_result_Series["fp"] = result_Series["fp"]
                best_result_Series["tn"] = result_Series["tn"]
                best_result_Series["fn"] = result_Series["fn"]
                best_result_Series["best_thresh_oi"] = thresh_oi
                best_result_Series["best_thresh_lvo"] = thresh_lvo

                best_result_Hemi["acc"] = result_Hemi["acc"]
                best_result_Hemi["sen"] = result_Hemi["sen"]
                best_result_Hemi["spe"] = result_Hemi["spe"]
                best_result_Hemi["auc"] = result_Hemi["auc"]
                best_result_Hemi["tp"] = result_Hemi["tp"]
                best_result_Hemi["fp"] = result_Hemi["fp"]
                best_result_Hemi["tn"] = result_Hemi["tn"]
                best_result_Hemi["fn"] = result_Hemi["fn"]
                best_result_Hemi["best_thresh_oi"] = thresh_oi
                best_result_Hemi["best_thresh_lvo"] = thresh_lvo


    return best_result_Series, best_result_Hemi


# 최적값 구하기
best_result_Series, best_result_Hemi = find_optimal_thresholds(merged_df, branch_by_OI=True, optimal_by_Series=True)

Series_acc_list.append(best_result_Series["acc"])
Series_sen_list.append(best_result_Series["sen"])
Series_spe_list.append(best_result_Series["spe"])
Series_auc_list.append(best_result_Series["auc"])
Hemi_acc_list.append(best_result_Hemi["acc"])
Hemi_sen_list.append(best_result_Hemi["sen"])
Hemi_spe_list.append(best_result_Hemi["spe"])
Hemi_auc_list.append(best_result_Hemi["auc"])

print('\n### 3. Branch by OI (optimize Threshold)')
print('3-1. Series 기준')
print(f'Best Accuracy : {best_result_Series["acc"]:.4f}')
print(f'Best Sensitivity : {best_result_Series["sen"]:.4f}')
print(f'Best Specificity : {best_result_Series["spe"]:.4f}')
print(f'Best AUC : {best_result_Series["auc"]:.4f}')
print(f'Best TP : {best_result_Series["tp"]}, FP : {best_result_Series["fp"]}, '
      f'TN : {best_result_Series["tn"]}, FN : {best_result_Series["fn"]}')
print(f'Best threshold for OI: {best_result_Series["best_thresh_oi"]:.1f}')
print(f'Best threshold for LVO: {best_result_Series["best_thresh_lvo"]:.1f}')
print('3-2. Hemi 기준')
print(f'Best Accuracy : {best_result_Hemi["acc"]:.4f}')
print(f'Best Sensitivity : {best_result_Hemi["sen"]:.4f}')
print(f'Best Specificity : {best_result_Hemi["spe"]:.4f}')
print(f'Best AUC : {best_result_Hemi["auc"]:.4f}')
print(f'Best TP : {best_result_Hemi["tp"]}, FP : {best_result_Hemi["fp"]}, '
      f'TN : {best_result_Hemi["tn"]}, FN : {best_result_Hemi["fn"]}')
print(f'Best threshold for OI: {best_result_Hemi["best_thresh_oi"]:.1f}')
print(f'Best threshold for LVO: {best_result_Hemi["best_thresh_lvo"]:.1f}')



best_result_Series, best_result_Hemi = find_optimal_thresholds(merged_df, branch_by_OI=False, optimal_by_Series=True)

Series_acc_list.append(best_result_Series["acc"])
Series_sen_list.append(best_result_Series["sen"])
Series_spe_list.append(best_result_Series["spe"])
Series_auc_list.append(best_result_Series["auc"])
Hemi_acc_list.append(best_result_Hemi["acc"])
Hemi_sen_list.append(best_result_Hemi["sen"])
Hemi_spe_list.append(best_result_Hemi["spe"])
Hemi_auc_list.append(best_result_Hemi["auc"])

print('\n### 4. Do not use OI (optimize Threshold)')
print('4-1. Series 기준')
print(f'Best Accuracy : {best_result_Series["acc"]:.4f}')
print(f'Best Sensitivity : {best_result_Series["sen"]:.4f}')
print(f'Best Specificity : {best_result_Series["spe"]:.4f}')
print(f'Best AUC : {best_result_Series["auc"]:.4f}')
print(f'Best TP : {best_result_Series["tp"]}, FP : {best_result_Series["fp"]}, '
      f'TN : {best_result_Series["tn"]}, FN : {best_result_Series["fn"]}')
print(f'Best threshold for OI: {best_result_Series["best_thresh_oi"]:.1f}')
print(f'Best threshold for LVO: {best_result_Series["best_thresh_lvo"]:.1f}')
print('4-2. Hemi 기준')
print(f'Best Accuracy : {best_result_Hemi["acc"]:.4f}')
print(f'Best Sensitivity : {best_result_Hemi["sen"]:.4f}')
print(f'Best Specificity : {best_result_Hemi["spe"]:.4f}')
print(f'Best AUC : {best_result_Hemi["auc"]:.4f}')
print(f'Best TP : {best_result_Hemi["tp"]}, FP : {best_result_Hemi["fp"]}, '
      f'TN : {best_result_Hemi["tn"]}, FN : {best_result_Hemi["fn"]}')
print(f'Best threshold for OI: {best_result_Hemi["best_thresh_oi"]:.1f}')
print(f'Best threshold for LVO: {best_result_Hemi["best_thresh_lvo"]:.1f}')



def plot_metric(acc_list, sen_list, spe_list, auc_list, title):
    # x축을 지표 이름으로, y축을 성능으로 하는 막대 그래프를 그립니다.
    x_labels = ['Branch by OI', 'Do not use OI', 'Branch by OI (optimized)', 'Do not use OI (optimized)']

    # Adjust the figure size
    plt.figure(figsize=(20, 12))

    # 각 지표별로 막대 그래프를 그립니다.
    x_positions = np.arange(len(x_labels))
    width = 0.2
    plt.bar(x_positions - 1.5*width, acc_list, width=width, label='Accuracy', alpha=0.5)
    plt.bar(x_positions - 0.5*width, sen_list, width=width, label='Sensitivity', alpha=0.5)
    plt.bar(x_positions + 0.5*width, spe_list, width=width, label='Specificity', alpha=0.5)
    plt.bar(x_positions + 1.5*width, auc_list, width=width, label='AUC', alpha=0.5)

    # 각 막대 그래프 위에 해당하는 값을 표시합니다.
    for i in range(len(x_labels)):
        plt.text(x=i-1.5*width-0.1, y=acc_list[i] + 0.02, s=f'{acc_list[i]:.2f}')
        plt.text(x=i-0.5*width-0.1, y=sen_list[i] + 0.02, s=f'{sen_list[i]:.2f}')
        plt.text(x=i+0.5*width-0.1, y=spe_list[i] + 0.02, s=f'{spe_list[i]:.2f}')
        plt.text(x=i+1.5*width-0.1, y=auc_list[i] + 0.02, s=f'{auc_list[i]:.2f}')

    # 그래프 제목과 범례, y축 레이블을 추가합니다.
    plt.title(title)
    plt.xlabel('Performance Indicator')
    plt.ylabel('Performance')
    plt.legend()

    # x축 눈금 레이블을 지정합니다.
    plt.xticks(x_positions, x_labels)

    # Rotate x-axis labels
    # plt.xticks(rotation=45)

    plt.show()

plot_metric(acc_list=Series_acc_list, sen_list=Series_sen_list, spe_list=Series_spe_list, auc_list=Series_auc_list, title='Performance - Series')
plot_metric(acc_list=Hemi_acc_list, sen_list=Hemi_sen_list, spe_list=Hemi_spe_list, auc_list=Hemi_auc_list, title='Performance - Hemi')