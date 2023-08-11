# confusion matrix(분류 결과표): 타겟의 원래 클래스와 예측한 클래스가 일치하는지를 갯수로 센 결과를 표로 나타낸것
# 정답 클래스를 행으로 / 예측 클래스를 열으로 나타냄
from sklearn.metrics import confusion_matrix
y_true = [2,0,2,2,0,1]
y_pred = [0,0,2,2,0,2]
confusion_matrix(y_true, y_pred)

# 이중 특히 양성과 음성 2가지로 표현한것을 이진 분류결과표라고함 (Binary Confusion Matrix)
#         예측양성  예측음성
# 실제 양성   TP      FN
# 실제 음성   FP      TN

y_true = [1,0,1,1,0,1]
y_pred = [0,0,1,1,0,1]
confusion_matrix(y_true, y_pred, labels=[1,0])

# Accuracy 정확도
# 전체 샘플중 맞게 예측한 샘플수
# (TP + TN) / Total
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)

# Precision 정밀도
# 양성이라고 판단한 샘플중 실제 양성 클래스인것 => 진짜라고 했는데 진짜 => 저격수(맞췄다 생각한거중 맞춘확률)
# TP / (TP + FP)
from sklearn.metrics import precision_score
precision_score(y_true, y_pred, pos_label=1)

# recall (재현율) or sensitivity(민감도) or TPR(True Positive Rate)
# 실제 양성중에 양성으로 예측된 비율
# TP / (TP + FN)
from sklearn.metrics import recall_score
# pos_label에 기준을 적어줌 1 or 'yes'
recall_score(y_true, y_pred, pos_label=1)

# 위 값들은 실제로 1이 많아 그냥 모두 1이라고 예측할 경우 정확도가 높아지고 precision이 높아지는 경우가 있어 그나마 조금더 수치를 조정하여 만든것이 F Score
# f1 score
# (1+b^n) * (precision*recall) / (precision + recall)
# 가중치가 1인 경우를 F1 Score 라고 함
# 정밀도와 재현율의 가중 조화평균
from sklearn.metrics import f1_score
print(f1_score(y_true, y_pred, pos_label=1))

# specificity(특이도)
# 실제 음성중에 음성으로 예측된 비율
# TN / (TN + FP)
# 1- FPR

# FPR (False Positive Rate)
# 실제 음성인데 양성이라고 잘못 예측된 비율
# FP / TN + FP
# 1 - 특이도

# TPR 과 FPR 은 서로 비례 => 조금만 1일꺼 같아도 1이라고 판정하는 모델은 TPR , FPR 함께상승. => 반대도 동일
# 단점, 잘예측한 TPR 높이면 잘못예측한 FPR 또한 높아짐 => 따라서 FPR 대비 TPR 좋게 나오도록 => 시각화 한것이 ROC 커브

# ROC curve (Receiver Operating Characteristics)
# 장점 : 면적을 측정하여 TPR FPR 복합적으로 평가가능
# 면적을 area under the curve:AUC 라고 하고 1에 가까울수록 성능이 좋고, 0.5에 가까울수록 성능이 나쁨
# FPR , TPR , thresholds 해석
# case1 의사가 모든 환자들을 암 확률이 어느정도만 되어도(threshold가 낮음) 암 환자로 판정하면 TPR(실제 암이걸린환자를 암이라 판정) 과 FPR(암이걸리지 않았지만 암이라 판정) 이 함께 높아짐
# => threshold 낮다 => TPR & FPR 비율 높음
# case2 의사가 모든 환자들을 암 확률이 매우높아야만(threshold가 높음) 암 환자로 판정하면 TPR(실제 암이걸린환자를 암이라 판정) 과 FPR(암이걸리지 않았지만 암이라 판정) 이 함께 낮아진다.
# => threshold 높다 => TPR & FPR 비율 낮음
# 따라서 threshold 를 낮출수록 TPR & FPR 비율이 높아지고, threshold 를 높일수록 TPR & FPR 비율이 낮아진다.
# 그리고 보통 FPR, TPR 순서로 확률 분포가 위치하여 threshold를 낮추면 TPR이 먼저 증가하고, threshold를 높이면 FPR이 먼저 사라진다.
from sklearn.metrics import roc_curve
# thresholds : 임계값 => 의미: thresholds 의 이하의 값은 False, 이상은 True 로 판단.
# 이 임계값에 따라 ROC 커브 위의 점 위치가 변화한다.
# 휨 정도가 크다는것은 == 면적이 1에 가깝다는것은 두 클래스를 더욱 잘 구별 할수 있다는것이다. 즉, 아래 면적이 높을수록 좋다.
fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
print(fpr, tpr, thresholds)

# 면적 구하는법
# AUC : 아래 면적이 1에 가까울수록, 넓을 수록 좋은 모형
from sklearn.metrics import auc
auc(fpr, tpr)

# 데이터 정답과 예측으로 바로 auc 구하는법
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_pred)

# 성능평가 모두 구하기
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))