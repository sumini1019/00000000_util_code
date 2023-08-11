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
#
pd.set_option('display.precision', 4)

# df = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20221027_new_cELVO_LVO_LSTM 결과\전체데이터 결과_Model#1_Weight#1.csv')
df = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20221027_new_cELVO_LVO_LSTM 결과\★전체데이터 결과_Model#1_Weight#3 (EIC, DMS, OI Prob 포함).csv')


# df = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\2_cELVO(LSTM)\20221027_new_cELVO_LVO_LSTM 결과\전체데이터 결과_Model#1_Weight#6.csv')

label = df['Label_LVO']
prob = df['Prob_LVO']

th = 0.5

def plot_roc_curve(fper, tper, auc_score):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve : AUC ({auc_score:.4f})')
    plt.legend()
    plt.show()

def get_cf_matrix(df, threshold):
    TP_total = 0
    TN_total = 0
    FP_total = 0
    FN_total = 0

    for i in range(len(df)):
        cur_label = df['Label_LVO'][i]
        cur_pred = df['Prob_LVO'][i]

        # 초기화
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        if cur_label == 1 and (cur_pred >= threshold):
            TP = 1
        elif cur_label == 1 and (cur_pred < threshold):
            FN = 1
        elif cur_label == 0 and (cur_pred >= threshold):
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

# AUC 계산
roc_auc = roc_auc_score(label, prob)
print("ROC AUC Score: {}".format(roc_auc))

# ROC Curve 그리기
fper, tper, thresholds = roc_curve(label, prob)
plot_roc_curve(fper, tper, roc_auc)

# Confusion Matrix 계산
result_CM = get_cf_matrix(df, threshold=th)
print(result_CM)

def plot_threshold_metrics(y_true, y_pred_prob):
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
    ax.set_title('Metrics by Threshold')
    ax.legend(loc='best')
    plt.show()

plot_threshold_metrics(label, prob)

def get_Result(df, name_inst='', th=0.0):

    eps = 1e-10

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # 현재 Threshold에 맞게, CM 새로 저장
    CM_new = []

    df_cur_Inst = df[df['Patient_ID'].cur_folder.contains(name_inst)].reset_index(drop=True)

    for index, row in df_cur_Inst.iterrows():
        # Confusion Matrix 생성
        if row['Label_LVO'] == 1 and row['Prob_LVO'] >= th:
            CM_new.append('TP')
            tp += 1
        elif row['Label_LVO'] == 1 and row['Prob_LVO'] < th:
            CM_new.append('FN')
            fn += 1
        elif row['Label_LVO'] == 0 and row['Prob_LVO'] < th:
            CM_new.append('TN')
            tn += 1
        elif row['Label_LVO'] == 0 and row['Prob_LVO'] >= th:
            CM_new.append('FP')
            fp += 1
        else:
            sys.exit('Error - CM 정할 수 없음')

    print('* Inst - {}'.format(name_inst))
    print('Accuracy: {:.4f}. Precision: {:.4f}. Recall: {:.4f}. Specificity: {:.4f}. AUC: {:.4f}'.format(
        (tp + tn) / (tp + tn + fp + fn + eps), tp / (tp + fp + eps), tp / (tp + fn + eps), tn / (tn + fp + eps),
        roc_auc_score(df_cur_Inst['Label_LVO'], df_cur_Inst['Prob_LVO'])))
    print('TP: {}    TN: {}    FP: {}    FN: {}'.format(tp, tn, fp, fn))

    # 기존 컬럼명 변경
    df_cur_Inst = df_cur_Inst.rename(columns={'CM': 'CM_Th0.5'})
    # CM 새로 삽입
    df_cur_Inst['CM_Th{}'.format(th)] = CM_new

    # 성능 Dictionary
    dict_result = {'Inst.': name_inst,
                   'Accuracy': '{:.4f}'.format((tp + tn) / (tp + tn + fp + fn + eps)),
                   'Precision': '{:.4f}'.format(tp / (tp + fp + eps)),
                   'Sensitivity': '{:.4f}'.format(tp / (tp + fn + eps)),
                   'Specificity': '{:.4f}'.format(tn / (tn + fp + eps)),
                   'AUC': '{:.4f}'.format(roc_auc_score(df_cur_Inst['Label_LVO'], df_cur_Inst['Prob_LVO'])),
                   'Num_Sample': len(df_cur_Inst)
                   }

    # DataFrame 결과 저장
    df_cur_Inst.to_csv('Result_Inst_{}_Th_{}.csv'.format(name_inst, th), index=False)

    return df_cur_Inst, dict_result

# 기관 별, 성능 산출
df_AJU, result_AJU = get_Result(df, name_inst='HR-A', th=th)
df_G, result_G = get_Result(df, name_inst='HR-G', th=th)
df_EU, result_EU = get_Result(df, name_inst='HR-E', th=th)
df_CSU, result_CSU = get_Result(df, name_inst='HR-C', th=th)
df_CNU, result_CNU = get_Result(df, name_inst='HR-S', th=th)
df_SCH, result_SCH = get_Result(df, name_inst='HR-B', th=th)
df_M, result_M = get_Result(df, name_inst='HR-M', th=th)

# 결과 Dict 묶기
df_dict = pd.DataFrame()
df_dict = df_dict.append(result_AJU, ignore_index=True)
df_dict = df_dict.append(result_G, ignore_index=True)
df_dict = df_dict.append(result_EU, ignore_index=True)
df_dict = df_dict.append(result_CSU, ignore_index=True)
df_dict = df_dict.append(result_CNU, ignore_index=True)
df_dict = df_dict.append(result_SCH, ignore_index=True)
df_dict = df_dict.append(result_M, ignore_index=True)

# 컬럼명 재정렬
df_dict = df_dict[['Inst.', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'AUC', 'Num_Sample']]

# 결과 저장
df_dict.to_csv('Summary_All-Inst_Th_{}.csv'.format(th), index=False)

print(df_dict)