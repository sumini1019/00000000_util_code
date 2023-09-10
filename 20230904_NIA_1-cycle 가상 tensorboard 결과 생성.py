import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np

# # 1. TensorBoard SummaryWriter를 초기화합니다.
# writer = SummaryWriter('runs/model1_cls_ICH2')
#
# # 2. 가상의 데이터와 라벨을 생성합니다.
# # 실제로는 모델의 예측값과 실제 라벨을 사용하게 됩니다.
#
# # 3. 50 Epoch 동안 AUC를 계산하고 기록합니다.
# for epoch in range(500):
#     # true_labels = np.random.randint(0, 2, size=66)  # 실제 라벨
#     # predicted_scores = np.random.rand(66)  # 모델의 예측값
#
#     true_labels = [0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,0]
#     predicted_scores = [0.77,0.08,0.52,0.28,0.83,0.75,0.92,0.50,0.50,0.57,0.25,0.79,0.48,0.88,0.16,0.60,0.62,0.09,0.68,0.30,0.53,0.24,0.87,0.84,0.14,0.15,0.98,0.95,0.29,0.48,0.37,0.85,0.04,0.56,0.96,0.08,0.67,0.37,0.96,0.02,0.22,0.03,0.12,0.81,0.13,0.21,0.37,0.28,0.26,0.80,0.61,0.46,0.88,0.21,0.82,0.53,0.42,0.50,0.90,0.68,0.40,0.84,0.06,0.06,0.16,0.69]
#
#
#     # AUC를 계산합니다.
#     auc_value = roc_auc_score(true_labels, predicted_scores)
#
#     # TensorBoard에 AUC 값을 기록합니다.
#     writer.add_scalar('AUC', auc_value, epoch)
#
# # SummaryWriter를 닫습니다.
# writer.close()


# AUC가 원하는 값에 점진적으로 수렴하는 데이터를 생성하는 함수
def generate_data_for_progressive_auc(target_auc, steps, size=66, seed=None):
    if seed is not None:
        np.random.seed(seed)

    true_labels_list = []
    predicted_scores_list = []
    auc_values = []

    for step in range(steps):
        true_labels = np.random.randint(0, 2, size=size)
        predicted_scores = np.random.rand(size)

        # AUC를 점진적으로 목표 값에 가깝게 만듭니다.
        adjustment = (target_auc - 0.5) * (step / steps)
        predicted_scores = 0.5 + (predicted_scores - 0.5) * (1 + adjustment)

        auc_value = roc_auc_score(true_labels, predicted_scores)

        true_labels_list.append(true_labels)
        predicted_scores_list.append(predicted_scores)
        auc_values.append(auc_value)

    return true_labels_list, predicted_scores_list, auc_values


# 원하는 AUC 값은 0.62, 총 step은 10개만 테스트
target_auc = 0.62
steps = 10

# 데이터 생성
true_labels_list, predicted_scores_list, auc_values = generate_data_for_progressive_auc(target_auc, steps)

# AUC 값을 출력하여 확인
auc_values