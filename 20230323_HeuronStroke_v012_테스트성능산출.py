import numpy as np
import pandas as pd
from module_sumin.utils_sumin import *
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score

# df = read_csv_autodetect_encoding(r'Z:\Sumin_Jung\00000000_RESULT\4_cSTROKE\20230323_HeuronStroke_v012_ICH양성, LVO양성, 정상 테스트셋 테스트\Result_cSTROKE_99.csv')
df = read_csv_autodetect_encoding(r'C:\Users\user\Downloads\test_lvo_low.csv')
df = pd.read_excel(r'Z:\Sumin_Jung\00000000_RESULT\1_cHS\20220120_cHS LSTM 결과 Validation (RSNA 데이터 n100)\cHS LSTM_RSNA 테스트_5epoch.xlsx')
def binary_classification_performance_byLabel(y_true, y_pred_Label):
    # Confusion matrix 계산
    tp = np.sum(np.logical_and(y_pred_Label == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred_Label == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred_Label == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred_Label == 0, y_true == 1))

    # 정확도(Accuracy), 민감도(Sensitivity), 특이도(Specificity) 계산
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # 결과 반환
    return accuracy, sensitivity, specificity, tp, tn, fp, fn

def binary_classification_performance_byProb(y_true, y_pred, threshold=0.5):
    predictions = [1 if prob >= threshold else 0 for prob in y_pred]

    # 혼동 행렬로부터 TP, TN, FP, FN 계산
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # 정확도, 민감도, 특이도 계산
    accuracy = accuracy_score(y_true, predictions)
    sensitivity = recall_score(y_true, predictions)
    specificity = tn / (tn + fp)

    print(f"Accuracy: {accuracy:0.4f}")
    print(f"Sensitivity: {sensitivity:0.4f}")
    print(f"Specificity: {specificity:0.4f}")

    return accuracy, sensitivity, specificity, tp, tn, fp, fn

# y_true: 정답값, y_pred: 예측값
# accuracy, sensitivity, specificity, tp, tn, fp, fn = binary_classification_performance(y_true=df['GT_LVO'], y_pred=df['cELVO_Result_Bi'])

accuracy, sensitivity, specificity, tp, tn, fp, fn = binary_classification_performance_byProb(y_true=df['Label'], y_pred=df['Probability'], threshold=0.8)
series_AUC = plot_roc_curve(df['Label'], df['Probability'], 'LVO (Series)', True)

accuracy, sensitivity, specificity, tp, tn, fp, fn = binary_classification_performance_byProb(y_true=df['Hemo_Series'], y_pred=df['Pred_5'], threshold=0.5)
series_AUC = plot_roc_curve(df['Hemo_Series'], df['Pred_5'], 'Heuron_ICH Internal Test', True)


print('\n1. Heuron-ELVO Eval. (Series-wise)')
print(f'Total : {tp+tn+fp+fn}')
print(f'TP : {tp}')
print(f'TN : {tn}')
print(f'FP : {fp}')
print(f'FN : {fn}')
print(f'Acc : {accuracy:0.4f}')
print(f'Sen : {sensitivity:0.4f}')
print(f'Spe : {specificity:0.4f}')
print(f'AUC : {series_AUC:0.4f}')

accuracy, sensitivity, specificity, tp, tn, fp, fn = binary_classification_performance_byLabel(y_true=df['GT_ICH'], y_pred_Label=df['ICH_Result'])

print('\n2. Heuron-ICH Eval.')
print(f'Total : {tp+tn+fp+fn}')
print(f'TP : {tp}')
print(f'TN : {tn}')
print(f'FP : {fp}')
print(f'FN : {fn}')
print(f'Acc : {accuracy:0.4f}')
print(f'Sen : {sensitivity:0.4f}')
print(f'Spe : {specificity:0.4f}')