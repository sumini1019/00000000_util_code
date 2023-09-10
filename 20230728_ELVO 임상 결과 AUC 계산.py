# 2023.07.28
# - ELVO 임상 결과를 기반으로 ROC Curve, AUC 계산
# - Whole Brain / Hemisphere 기준으로 각각 계산함
#
# 2023.08.16
# - AUC의 CI를 계산하기 위해서, 부트스트랩을 추가함
# - 원래 샘플에서 중복 허용해서 리샘플링하고, AUC 계산을 반복한 후 (default : 1000회) CI를 계산함

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r'C:\Users\user\Downloads\elvo_clinic_test_2.csv')

# Create the new label columns
df['Label_LH'] = (df['Label'] == 'LH').astype(int)
df['Label_RH'] = (df['Label'] == 'RH').astype(int)
df['Label_binary'] = (df['Label'] != 'Normal').astype(int)

# Replace ' - ' with 0 and convert the columns to float
for col in [' LVO_WB ', ' LVO_prob_Left ', ' LVO_prob_Right ']:
    df[col] = df[col].replace(' - ', 0).astype(float)

# Function to calculate AUC's 95% CI using bootstrap
def compute_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.05):
    bootstrapped_aucs = []
    for _ in range(n_bootstraps):
        y_true_boot, y_score_boot = resample(y_true, y_score)
        fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
        auc_boot = auc(fpr_boot, tpr_boot)
        bootstrapped_aucs.append(auc_boot)

    lower_bound = max(0.0, np.percentile(bootstrapped_aucs, alpha / 2 * 100))
    upper_bound = min(1.0, np.percentile(bootstrapped_aucs, (1 - alpha / 2) * 100))

    return lower_bound, upper_bound

# Function to calculate and plot ROC Curve and AUC
def plot_roc_curve(y_true, y_score, label, is_hemisphere):
    for i in range(3):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        lower_auc, upper_auc = compute_auc_ci(y_true, y_score)
        plt.figure()
        num_sample = len(y_true)//2 if is_hemisphere else len(y_true)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (%s)\n(AUC = %0.4f) (95%% CI: %0.4f-%0.4f) (n=%d)' % (label, roc_auc, lower_auc, upper_auc, num_sample))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

# Whole Brain
y_score_wholebrain = df[' LVO_WB ']
y_true_wholebrain = df['Label_binary']
plot_roc_curve(y_true_wholebrain, y_score_wholebrain, 'Whole Brain', is_hemisphere=False)

# Hemisphere (Left and Right combined)
y_score_hemisphere = pd.concat([df[' LVO_prob_Left '], df[' LVO_prob_Right ']], ignore_index=True)
y_true_hemisphere = pd.concat([df['Label_LH'], df['Label_RH']], ignore_index=True)
plot_roc_curve(y_true_hemisphere, y_score_hemisphere, 'Hemisphere', is_hemisphere=True)
