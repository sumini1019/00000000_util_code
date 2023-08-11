### 목적 ###
# -> 환자의 슬라이스 별 probability 값을 사용하여,
# -> 환자 별 dms 여부 결정

import pandas as pd
import os

# 임계값 - 슬라이스의 dms 여부
threshold_dms = 0.9
# 임계값 - 전체 슬라이스 대비, dms 슬라이스의 비율
threshold_rate_dms = 0.25
# 임계값 - 전체 슬라이스 대비, dms 슬라이스 개수 (임계값 이상 시, dms)
threshold_num_dms = 1

# 환자 번호 없는 리스트
list_no_patient = []

# 전체 환자에 대한, 성능 카운트
TP_all = 0
TN_all = 0
FP_all = 0
FN_all = 0

path_csv = 'Z:/Sumin_Jung/00000000_RESULT/2_cELVO/20210406_Vessel에 대한 DMS Classification 결과 정리/10. Half_v2 데이터 사용하여, embedding 뽑으면서 학습/fold1(개선 후)'

# csv 데이터 프레임 로드
df_probability = pd.read_csv(os.path.join(path_csv, 'Result_Label_and_Prob.csv'))


# 환자 별 결과 저장할 데이터 프레임 생성
# df_result = pd.DataFrame({"Patient_ID", "Prob_DMS_Left", "Prob_DMS_Right", "Prob_EIC_Left", "Prob_EIC_Right", "Prob_LVO_Left", "Prob_LVO_Right"})
df_result = pd.DataFrame(columns={"Patient_ID", "Num_Total_Slice",
                                  "GT_Right", "GT_Left",
                                  "Num_DMS_Slice_Right", "Num_DMS_Slice_Left",
                                  "Patient_DMS_Right", "Patient_DMS_Left",
                                  "TP_Right", "TN_Right", "FP_Right", "FN_Right",
                                  "TP_Left", "TN_Left", "FP_Left", "FN_Left",
                                  "Patient_Diagnosis"})

# 환자 번호 수
num_patient = max(df_probability['ID'])

df_probability = df_probability[['ID', 'ID_Filename', 'DMS - R', 'DMS - L', 'Pred_DMS_R_Quarter', 'Pred_DMS_L_Quarter']]

# 환자 번호 순으로 반복
for i in range(1, num_patient+1):
    # 변수 초기화
    TP_right = 0
    TN_right = 0
    FP_right = 0
    FN_right = 0
    TP_left = 0
    TN_left = 0
    FP_left = 0
    FN_left = 0

    # 현재 환자의 데이터 파싱
    df_cur_patient = df_probability[df_probability['ID'] == i]

    # 해당 환자 번호의 데이터가 없을 경우 다음 번호로 넘어갈 것
    if len(df_cur_patient) == 0:
        list_no_patient.append(i)
        continue

    # 환자 별 정답값 생성
    # - Left / Right 별로 DMS가 하나라도 있을 경우, 환자의 DMS 결과는 True
    if sum(df_cur_patient['DMS - R']) > 0:  # right
        label_patient_dms_right = True
    else:
        label_patient_dms_right = False
    if sum(df_cur_patient['DMS - L']) > 0:  # left
        label_patient_dms_left = True
    else:
        label_patient_dms_left = False

    ###############################################################################
    # 아래는 모두 Result 관련
    ###############################################################################

    # Probability가 threshold 넘는 slice의 개수 계산
    num_slice_dms_right = list(df_cur_patient['Pred_DMS_R_Quarter'] > threshold_dms).count(True)
    num_slice_dms_left = list(df_cur_patient['Pred_DMS_L_Quarter'] > threshold_dms).count(True)

    # 환자의 threshold 넘는 slice의 개수가, 비율 threshold를 넘는지 확인
    # rate_dms_right = (num_slice_dms_right / len(df_cur_patient))
    # rate_dms_left = (num_slice_dms_left / len(df_cur_patient))

    # 비율 threshold 결과에 따라, 환자 정상/dms 결정
    # if rate_dms_right >= threshold_rate_dms:
    if num_slice_dms_right >= threshold_num_dms:
        patient_dms_right = True
    else:
        patient_dms_right = False
    # if rate_dms_left >= threshold_rate_dms:
    if num_slice_dms_left >= threshold_num_dms:
        patient_dms_left = True
    else:
        patient_dms_left = False

    # 환자의 Diagnosis 결정
    # - Normal / Left / Right / Bileteral
    if patient_dms_right==True and patient_dms_left==True:
        patient_diagnosis = "Bileteral"
    elif patient_dms_right==True and patient_dms_left==False:
        patient_diagnosis = "Right"
    elif patient_dms_right == False and patient_dms_left == True:
        patient_diagnosis = "Left"
    else:
        patient_diagnosis = "Normal"

    # 정답과, 모델 결과의 일치 판정
    if label_patient_dms_right==True and patient_dms_right==True:
        TP_right = 1
    elif label_patient_dms_right==True and patient_dms_right==False:
        FN_right = 1
    elif label_patient_dms_right == False and patient_dms_right == True:
        FP_right = 1
    else:
        TN_right = 1

    if label_patient_dms_left==True and patient_dms_left==True:
        TP_left = 1
    elif label_patient_dms_left==True and patient_dms_left==False:
        FN_left = 1
    elif label_patient_dms_left == False and patient_dms_left == True:
        FP_left = 1
    else:
        TN_left = 1

    # 현재 환자 결과 저장
    df_result = df_result.append(
        {"Patient_ID": i, "Num_Total_Slice": len(df_cur_patient),
         "GT_Right":label_patient_dms_right, "GT_Left":label_patient_dms_left,
         "Num_DMS_Slice_Right": num_slice_dms_right, "Num_DMS_Slice_Left": num_slice_dms_left,
         "Patient_DMS_Right": patient_dms_right, "Patient_DMS_Left": patient_dms_left,
         "TP_Right": TP_right, "TN_Right": TN_right, "FP_Right": FP_right, "FN_Right": FN_right,
         "TP_Left": TP_left, "TN_Left": TN_left, "FP_Left": FP_left, "FN_Left": FN_left,
         "Patient_Diagnosis":patient_diagnosis
         }, ignore_index=True)

    # 전체 환자에 대한 성능 카운트 누적
    TP_all = TP_all + TP_right + TP_left
    TN_all = TN_all + TN_right + TN_left
    FP_all = FP_all + FP_right + FP_left
    FN_all = FN_all + FN_right + FN_left

# 컬럼 순서 재배치
df_result = df_result[["Patient_ID", "Num_Total_Slice",
                       "GT_Right", "GT_Left",
                       "Num_DMS_Slice_Right", "Num_DMS_Slice_Left",
                       "Patient_DMS_Right", "Patient_DMS_Left",
                       "TP_Right", "TN_Right", "FP_Right", "FN_Right",
                       "TP_Left", "TN_Left", "FP_Left", "FN_Left",
                       "Patient_Diagnosis"]]

# 최종 결과 저장
df_result.to_csv(os.path.join(path_csv, 'Result_Patient.csv'))

# 전체 환자에 대한, 지표 계산
result_num_case = TP_all + TN_all + FP_all + FN_all
result_accuracy = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
result_precision = TP_all / (TP_all + FP_all)
result_sensitivity = TP_all / (TP_all + FN_all)
result_specificity = TN_all / (TN_all + FP_all)

print("Num Case : ", result_num_case)
print("Accuracy : ", result_accuracy)
print("Precision : ", result_precision)
print("Sensitivity : ", result_sensitivity)
print("Specificity : ", result_specificity)