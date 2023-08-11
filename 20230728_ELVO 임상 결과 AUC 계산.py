import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r'C:\Users\user\Downloads\elvo_clinic_test_2.csv')

# Create the new label columns
df['Label_LH'] = (df['Label'] == 'LH').astype(int)
df['Label_RH'] = (df['Label'] == 'RH').astype(int)

# Create a binary label where LH and RH are 1, and Normal is 0
df['Label_binary'] = (df['Label'] != 'Normal').astype(int)

# Replace ' - ' with 0 and convert the columns to float
for col in [' LVO_WB ', ' LVO_prob_Left ', ' LVO_prob_Right ']:
    df[col] = df[col].replace(' - ', 0).astype(float)

# Calculate the ROC Curve and AUC based on LVO_WB
fpr, tpr, thresholds = roc_curve(df['Label_binary'], df[' LVO_WB '])
roc_auc = auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Whole Brain) (area = %0.4f) (n={len(df)})' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Calculate the ROC Curve and AUC based on LVO_prob_Left and LVO_prob_Right
y_score = pd.concat([df[' LVO_prob_Left '], df[' LVO_prob_Right ']], ignore_index=True)
y_true = pd.concat([df['Label_LH'], df['Label_RH']], ignore_index=True)
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Hemisphere) (area = %0.4f) (n={len(df)})' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
