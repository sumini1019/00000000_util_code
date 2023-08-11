# 2022.05.30 - AUC 만들자!
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\00000000 Code\20210203_cHS_SW_binary_segmentation_AND_classification_for_subtype\버전별 엔진\20210923_개발본부_전달버전 v0.02\all_result.csv')
# df = pd.read_csv(r'C:\Users\user\Downloads\20210805_성능정리_데이터 13개 추가.csv')

y_true = df['Label']
y_pred = df['subType_any']

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

def confusion_matrix(df, threshold_any):

    TP_total = 0
    TN_total = 0
    FP_total = 0
    FN_total = 0

    for i in range(len(df)):
        cur_label = df['Label'][i]
        cur_pred = df['subType_any'][i]

        # 초기화
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        if cur_label == 1 and (cur_pred >= threshold_any):
            TP = 1
        elif cur_label == 1 and (cur_pred < threshold_any):
            FN = 1
        elif cur_label == 0 and (cur_pred >= threshold_any):
            FP = 1
        else:
            TN = 1

        TP_total = TP_total + TP
        TN_total = TN_total + TN
        FP_total = FP_total + FP
        FN_total = FN_total + FN

    # 전체 환자에 대한, 지표 계산
    result_CM = {'num_case': TP_total + TN_total + FP_total + FN_total,
                 'num_TP': TP_total,
                 'num_TN': TN_total,
                 'num_FP': FP_total,
                 'num_FN': FN_total,
                 'accuracy': (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total),
                 'precision': TP_total / (TP_total + FP_total),
                 'sensitivity': TP_total / (TP_total + FN_total),
                 'specificity': TN_total / (TN_total + FP_total)}

    return result_CM

# ROC Curve 그리기
fper, tper, thresholds = roc_curve(y_true, y_pred)
plot_roc_curve(fper, tper)

# AUC 계산
roc_auc = roc_auc_score(y_true, y_pred)
print("ROC AUC Score: {}".format(roc_auc))

# Confusion Matrix 계산
result_CM = confusion_matrix(df, 0.7)
print(result_CM)

print('exit')