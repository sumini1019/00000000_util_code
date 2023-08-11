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

pd.set_option('display.precision', 4)

# df_result = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230220_ELVO_v012_5fold_cross_validation_결과(validation_n1000)\20230220_Result_LVO_Ensemble_Multi_Inference_Validation_n1000.csv')
df_result = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230220_ELVO_v012_5fold_cross_validation_결과(validation_n1000)\Result_LVO_Ensemble_Multi_Inference_until500.csv')
df_label = pd.read_csv(r'Z:/Stroke/DATA/DATA_cELVO_cASPECTS/Final_List_v2.csv')
# df_prob_fold = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20230220_ELVO_v012_5fold_cross_validation_결과(validation_n1000)\Result_LVO_Ensemble_Multi_Inference.csv')

# Dataframe merge
# - 7개 데이터 빠짐
# - ['HR-A-0374', 'HR-G-0876', 'HR-E-0038', 'HR-A-0438', 'HR-A-0332', 'HR-C-0032', 'HR-M-0009']
merged_df = pd.merge(df_result, df_label, left_on='DataPath', right_on='HR-ID')

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

# # OI 분기 결과
# pred_mean_CHo_LH = merged_df['Result_mean_LVO_LH']
# pred_mean_CHo_RH = merged_df['Result_mean_LVO_RH']
# pred_voting_CHo_LH = merged_df['Result_voting_LVO_LH']
# pred_voting_CHo_RH = merged_df['Result_voting_LVO_RH']
# prob_mean_CHo_LH = merged_df['Prob_LVO_LH']
# prob_mean_CHo_RH = merged_df['Prob_LVO_RH']
# label_LH = merged_df['LVO_L']
# label_RH = merged_df['LVO_R']

# 반구비교 + 반구비교X 개별 결과
pred_mean_CHo_LH = merged_df['Result_mean_LVO_CHo_LH']
pred_mean_CHo_RH = merged_df['Result_mean_LVO_CHo_RH']
pred_mean_CHx_LH = merged_df['Result_mean_LVO_CHx_LH']
pred_mean_CHx_RH = merged_df['Result_mean_LVO_CHx_RH']
pred_voting_CHo_LH = merged_df['Result_voting_LVO_CHo_LH']
pred_voting_CHo_RH = merged_df['Result_voting_LVO_CHo_RH']
pred_voting_CHx_LH = merged_df['Result_voting_LVO_CHx_LH']
pred_voting_CHx_RH = merged_df['Result_voting_LVO_CHx_RH']
pred_mean_CHo = pd.concat([pred_mean_CHo_LH, pred_mean_CHo_RH], ignore_index=True)
pred_mean_CHx = pd.concat([pred_mean_CHx_LH, pred_mean_CHx_RH], ignore_index=True)
pred_voting_CHo = pd.concat([pred_voting_CHo_LH, pred_voting_CHo_RH], ignore_index=True)
pred_voting_CHx = pd.concat([pred_voting_CHx_LH, pred_voting_CHx_RH], ignore_index=True)

label_LH = merged_df['LVO_L']
label_RH = merged_df['LVO_R']
label = pd.concat([label_LH, label_RH], ignore_index=True)

for fold in range(1, 6):
    prob_fold_CHo_LH = merged_df[f'preds_LVO_CHo_fold{fold}_LH']
    prob_fold_CHo_RH = merged_df[f'preds_LVO_CHo_fold{fold}_RH']
    prob_fold_CHx_LH = merged_df[f'preds_LVO_CHx_fold{fold}_LH']
    prob_fold_CHx_RH = merged_df[f'preds_LVO_CHx_fold{fold}_RH']
    prob_fold_CHo = pd.concat([prob_fold_CHo_LH, prob_fold_CHo_RH], ignore_index=True)
    prob_fold_CHx = pd.concat([prob_fold_CHx_LH, prob_fold_CHx_RH], ignore_index=True)

    def get_result_by_Pred(pred, label):
        TP = ((pred == 1) & (label == 1)).sum()
        TN = ((pred == 0) & (label == 0)).sum()
        FP = ((pred == 1) & (label == 0)).sum()
        FN = ((pred == 0) & (label == 1)).sum()

        acc = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)

        print(f"num_case: {TP + TN + FP + FN}")
        print(f"num_TP: {TP}")
        print(f"num_TN: {TN}")
        print(f"num_FP: {FP}")
        print(f"num_FN: {FN}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")

        return 1

    def get_result_by_Prob(prob, label, th=0.5):
        TP = ((prob >= th) & (label == 1)).sum()
        TN = ((prob < th) & (label == 0)).sum()
        FP = ((prob >= th) & (label == 0)).sum()
        FN = ((prob < th) & (label == 1)).sum()

        acc = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)

        print(f"num_case: {TP + TN + FP + FN}")
        print(f"num_TP: {TP}")
        print(f"num_TN: {TN}")
        print(f"num_FP: {FP}")
        print(f"num_FN: {FN}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")

        return 1

    ################################
    # 1. Mean 기준 성능
    ################################
    print(f'\n*** 반구비교O - Fold{fold} (Threshold = 0.5) ***')
    get_result_by_Prob(prob_fold_CHo, label, th=0.5)
    print(f'\n*** 반구비교X - Fold{fold} (Threshold = 0.5) ***')
    get_result_by_Prob(prob_fold_CHx, label, th=0.5)

    ################################
    # 3. Threshold 별 성능
    ################################
    def plot_threshold_metrics(y_true, y_pred_prob, CH=None, fold=None):
        """Threshold 0.1별로 Accuracy, Sensitivity, Specificity를 막대 그래프 형태로 보여주는 함수"""
        threshold_list = np.around(np.arange(0.1, 1.0, 0.1), decimals=3)
        accuracy_list, sensitivity_list, specificity_list = [], [], []
        for threshold in threshold_list:
            y_pred = (y_pred_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            accuracy_list.append(accuracy)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
        df = pd.DataFrame({
            'Threshold': threshold_list,
            'Accuracy': accuracy_list,
            'Sensitivity': sensitivity_list,
            'Specificity': specificity_list
        })
        df = df.set_index('Threshold')
        ax = df.plot(kind='bar', figsize=(10, 6))
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Metrics by Threshold - LVO_{CH} - Fold{fold}')
        ax.legend(loc='best')
        plt.show()

    # plot_threshold_metrics(label, prob_mean_CHo, CH='CHo', fold=fold)
    # plot_threshold_metrics(label, prob_mean_CHx, CH='CHx', fold=fold)

    ################################
    # 4. ROC, AUC 그리기 (Mean Ensemble)
    ################################
    def plot_roc_curve(fper, tper, auc_score, CH=None, fold=None):
        plt.plot(fper, tper, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve (LVO_{CH}) - AUC({auc_score:.4f}) - Fold{fold}")
        plt.legend()
        plt.show()

    # AUC 계산
    roc_auc = roc_auc_score(label, prob_fold_CHo)
    print(f"\n*** AUC Score CHo(Mean Probability) - Fold{fold} ***")
    print("AUC Score: {}".format(roc_auc))
    # ROC Curve 그리기
    fper, tper, thresholds = roc_curve(label, prob_fold_CHo)
    plot_roc_curve(fper, tper, roc_auc, CH='CHo', fold=fold)

    # AUC 계산
    roc_auc = roc_auc_score(label, prob_fold_CHx)
    print(f"*** AUC Score CHx(Mean Probability) - Fold{fold} ***")
    print("AUC Score: {}".format(roc_auc))
    # ROC Curve 그리기
    fper, tper, thresholds = roc_curve(label, prob_fold_CHx)
    plot_roc_curve(fper, tper, roc_auc, CH='CHx', fold=fold)

# 2023.02.21
# - 특정 fold만 조합해서 Ensemble 하도록 변경
use_fold = [1] #[1, 2, 4, 5]

# 5-0. Voting, Mean Ensemble
# Voting Ensemble
def voting(preds, use_fold, threshold=0.5):
    # 모델의 예측값에 대해 threshold를 적용하여, positive/negative로 분류
    # positive = 1, negative = 0
    # 분류값의 합 계산
    sum_classification = 0
    for i, pred in enumerate(preds):
        if i + 1 in use_fold:
            if pred >= threshold:
                sum_classification += 1
            else:
                sum_classification += 0

    # 앙상블 모델이 positive로 분류한 샘플의 비율이 0.5 이상이면 1로 설정, 그렇지 않으면 0으로 설정
    if sum_classification >= len(use_fold) / 2.0:
        result_voting = 1
    else:
        result_voting = 0

    # voting 결과
    return result_voting

# Mean Ensemble
def mean(preds, use_fold, threshold=0.5):
    # 사용할 fold 선택
    preds_select = []
    for i, pred in enumerate(preds):
        if i+1 in use_fold:
            preds_select.append(pred)

    # 예측값들의 평균 계산
    pred_mean = sum(preds_select) / len(preds_select)

    # mean 결과에 대한 thresholding
    if pred_mean >= threshold:
        result_mean = 1
    else:
        result_mean = 0

    return result_mean, pred_mean

# CHo, CHx 에 대해 확인
for CH in ['CHo', 'CHx']:
    # 5-1. 반구비교 Probability 기반 결과
    # 5-1-1. Fold별 Probability
    pred_esb_mean_LH = []
    pred_esb_mean_RH = []
    pred_esb_voting_LH = []
    pred_esb_voting_RH = []

    for idx, item in merged_df.iterrows():
        list_lvo_prob_LH = []
        list_lvo_prob_RH = []
        for i in range(5):
            list_lvo_prob_LH.append(item[f'preds_LVO_{CH}_fold{i+1}_LH'])
            list_lvo_prob_RH.append(item[f'preds_LVO_{CH}_fold{i+1}_RH'])
        # 5-1-2. Mean Ensemble
        th_esb_mean = 0.5
        result_esb_mean_LH, prob_mean_LH = mean(list_lvo_prob_LH, use_fold=use_fold, threshold=th_esb_mean)
        result_esb_mean_RH, prob_mean_RH = mean(list_lvo_prob_RH, use_fold=use_fold, threshold=th_esb_mean)
        # 5-1-3. Voting Ensemble
        th_esb_voting = 0.5
        result_esb_voting_LH = voting(list_lvo_prob_LH, use_fold=use_fold, threshold=th_esb_voting)
        result_esb_voting_RH = voting(list_lvo_prob_RH, use_fold=use_fold, threshold=th_esb_voting)
        # 5-1-4. Ensemble 결과 저장
        pred_esb_mean_LH.append(result_esb_mean_LH)
        pred_esb_mean_RH.append(result_esb_mean_RH)
        pred_esb_voting_LH.append(result_esb_voting_LH)
        pred_esb_voting_RH.append(result_esb_voting_RH)

    # 5-1-5. Label과 비교하여 성능 산출
    pred_esb_mean = pd.concat([pd.Series(pred_esb_mean_LH), pd.Series(pred_esb_mean_RH)], ignore_index=True)
    pred_esb_voting = pd.concat([pd.Series(pred_esb_voting_LH), pd.Series(pred_esb_voting_RH)], ignore_index=True)

    print(f'\n*** LVO_{CH} 모델 / Mean Ensemble (Threshold : {th_esb_mean}) ***')
    get_result_by_Pred(pred_esb_mean, label)
    print(f'\n*** LVO_{CH} 모델 / Voting Ensemble (Threshold : {th_esb_voting}) ***')
    get_result_by_Pred(pred_esb_voting, label)
