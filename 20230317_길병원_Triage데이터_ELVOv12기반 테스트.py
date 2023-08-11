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



df_result_prob = read_csv_autodetect_encoding(r'C:\Users\user\Downloads\Result_cSTROKE.csv')
df_label = read_csv_autodetect_encoding(r'C:\Users\user\Downloads\DA_Results_v221114_GIL.csv')

df_label['Patient_ID'] = df_label['Patient_ID'].str.strip("'")
merged_df = pd.merge(df_result_prob, df_label, left_on='Patient_ID', right_on='Patient_ID')

def eval_ELVO_performance_SeriesWise(merged_df, threshold_LVO=0.5):
    tp = fp = tn = fn = 0
    total_count = len(merged_df)

    for index, row in merged_df.iterrows():
        # 1. 두 방향 중 큰 Prob 계산
        max_prob = max(row['cELVO_Prob_Left'], row['cELVO_Prob_Right'])

        # 2. Label
        label = row['LVO_GT']

        # 3. TP, TN, FP, FN 계산
        if max_prob >= threshold_LVO and label == 1:
            tp += 1
        elif max_prob < threshold_LVO and label == 0:
            tn += 1
        elif max_prob >= threshold_LVO and label == 0:
            fp += 1
        else:
            fn += 1

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, sensitivity, specificity, tp, fp, tn, fn, total_count

# 각 성능 지표를 리스트에 저장합니다.
acc_list = []
sen_list = []
spe_list = []

th_LVO = 0.4
acc, sen, spe, tp, fp, tn, fn, total_count = eval_ELVO_performance_SeriesWise(merged_df, threshold_LVO=th_LVO)
acc_list.append(acc)
sen_list.append(sen)
spe_list.append(spe)

print(f'\n### 1. LVO (th : {th_LVO})')
print(f'Accuracy : {acc:.4f}')
print(f'Sensitivity : {sen:.4f}')
print(f'Specificity : {spe:.4f}')
print(f'TP : {tp}, FP : {fp}, TN : {tn}, FN : {fn}, Total : {total_count}')

def find_optimal_thresholds(merged_df):
    best_accuracy = 0
    best_sensitivity = 0
    best_specificity = 0
    best_tp = 0
    best_fp = 0
    best_tn = 0
    best_fn = 0
    best_thresh_oi = None
    best_thresh_lvo = None

    for thresh_lvo in np.arange(0.1, 1.0, 0.1):
        acc, sen, spe, tp, fp, tn, fn, total_count = eval_ELVO_performance_SeriesWise(merged_df,
                                                                           threshold_LVO=thresh_lvo)
        # 최적값 업데이트
        avg_metric = (acc + sen + spe) / 3
        if avg_metric > (best_accuracy + best_sensitivity + best_specificity) / 3:
            best_accuracy = acc
            best_sensitivity = sen
            best_specificity = spe
            best_tp = tp
            best_fp = fp
            best_tn = tn
            best_fn = fn
            best_thresh_lvo = thresh_lvo

    return best_thresh_lvo, best_accuracy, best_sensitivity, best_specificity, best_tp, best_fp, best_tn, best_fn


# 최적값 구하기
best_thresh_lvo, best_accuracy, best_sensitivity, best_specificity, best_tp, best_fp, best_tn, best_fn = \
    find_optimal_thresholds(merged_df)
acc_list.append(best_accuracy)
sen_list.append(best_sensitivity)
spe_list.append(best_specificity)

print('\n### 2. LVO (optimize Threshold)')
print(f'Best Accuracy : {best_accuracy:.4f}')
print(f'Best Sensitivity : {best_sensitivity:.4f}')
print(f'Best Specificity : {best_specificity:.4f}')
print(f'Best TP : {best_tp}, FP : {best_fp}, TN : {best_tn}, FN : {best_fn}')
print(f'Best threshold for LVO: {best_thresh_lvo:.1f}')