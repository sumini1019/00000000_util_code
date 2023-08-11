### 목적 ###
# -> 환자의 슬라이스 별 probability 값을 사용하여,
# -> 대표 probability를 계산하여 새로운 csv를 생성하는 코드

import shutil
import os
import pandas as pd

# 복사할 원본 파일 경로
# path = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Half/fold1'
# path = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Quarter'
# path_cur = ''
# 복사할 경로 - Case #1 (DMS)
# resultPath = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Half/fold1/dms'
# resultPath = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Quarter/fold1/dms'

### Hyper Parameter ###
# 슬라이스에서, DMS/EIC 여부 결정할 Probability 임계값
th_DMS_Slice_binary = 0.5
th_EIC_Slice_binary = 0.5
# DMS/EIC 조합 시, weight
weight_DMS = 0.7
weight_EIC = 0.3
# 환자 최종 LVO Probability에서, LVO 여부 결정할 Probability 임계값
th_LVO_Patient_binary = 0.5

# 환자 번호 없는 리스트
list_no_patient = []

# csv 데이터 프레임 로드
df_probability = pd.read_csv('Z:/Sumin_Jung/00000000_RESULT/2_cELVO/20210406_Vessel에 대한 DMS Classification 결과 정리/9. EIC 결합 반구 선택 데이터 사용/Probability_DMS_EIC.csv')

# 환자 별 결과 저장할 데이터 프레임 생성
# df_result = pd.DataFrame({"Patient_ID", "Prob_DMS_Left", "Prob_DMS_Right", "Prob_EIC_Left", "Prob_EIC_Right", "Prob_LVO_Left", "Prob_LVO_Right"})
df_result = pd.DataFrame(columns={"Patient_ID",
                       "Prob_DMS_Left_Half", "Prob_DMS_Right_Half",
                       "Prob_DMS_Left_Quarter", "Prob_DMS_Right_Quarter",
                       "Prob_EIC_Left", "Prob_EIC_Right",
                       "Prob_LVO_Left_Half", "Prob_LVO_Right_Half", "Binary_LVO_Right_Half", "Binary_LVO_Left_Half",
                       "Prob_LVO_Left_Quarter", "Prob_LVO_Right_Quarter", "Binary_LVO_Right_Quarter", "Binary_LVO_Left_Quarter"})

# 환자 번호 수
num_patient = max(df_probability['ID'])

# 환자 번호 순으로 반복
for i in range(1, num_patient+1):
    # 현재 환자의 Probability 파싱
    df_cur_patient = df_probability[df_probability['ID'] == i]

    # 해당 환자 번호의 데이터가 없을 경우 다음 번호로 넘어갈 것
    if len(df_cur_patient) == 0:
        list_no_patient.append(i)
        continue



    # DMS 사용할 데이터 파싱
    df_only_dms = df_cur_patient[df_cur_patient['USED_for_DMS'] == 1]
    # EIC 사용할 데이터 파싱 (전체니까 그대로 사용)
    df_only_eic = df_cur_patient

    # 1. DMS
    # 내림차순 정렬
    df_only_dms_right_Half = df_only_dms.sort_values(by=['Pred_DMS_R_Half'], axis=0, ascending=False)
    df_only_dms_left_Half = df_only_dms.sort_values(by=['Pred_DMS_L_Half'], axis=0, ascending=False)
    df_only_dms_right_Quarter = df_only_dms.sort_values(by=['Pred_DMS_R_Quarter'], axis=0, ascending=False)
    df_only_dms_left_Quarter = df_only_dms.sort_values(by=['Pred_DMS_L_Quarter'], axis=0, ascending=False)
    # Threshold 이상인 값만 파싱
    series_prob_dms_right_Half = df_only_dms_right_Half[df_only_dms_right_Half['Pred_DMS_R_Half'] > th_DMS_Slice_binary]
    series_prob_dms_left_Half = df_only_dms_left_Half[df_only_dms_left_Half['Pred_DMS_L_Half'] > th_DMS_Slice_binary]
    series_prob_dms_right_Quarter = df_only_dms_right_Quarter[df_only_dms_right_Quarter['Pred_DMS_R_Quarter'] > th_DMS_Slice_binary]
    series_prob_dms_left_Quarter = df_only_dms_left_Quarter[df_only_dms_left_Quarter['Pred_DMS_L_Quarter'] > th_DMS_Slice_binary]
    # 평균값
    if len(series_prob_dms_right_Half) > 0:
        rp_prob_dms_right_Half = sum(series_prob_dms_right_Half['Pred_DMS_R_Half']) / len(series_prob_dms_right_Half)
    else:
        rp_prob_dms_right_Half = 0
    if len(series_prob_dms_left_Half) > 0:
        rp_prob_dms_left_Half = sum(series_prob_dms_left_Half['Pred_DMS_L_Half']) / len(series_prob_dms_left_Half)
    else:
        rp_prob_dms_left_Half = 0

    if len(series_prob_dms_right_Quarter) > 0:
        rp_prob_dms_right_Quarter = sum(series_prob_dms_right_Quarter['Pred_DMS_R_Quarter']) / len(series_prob_dms_right_Quarter)
    else:
        rp_prob_dms_right_Quarter = 0
    if len(series_prob_dms_left_Quarter) > 0:
        rp_prob_dms_left_Quarter = sum(series_prob_dms_left_Quarter['Pred_DMS_L_Quarter']) / len(series_prob_dms_left_Quarter)
    else:
        rp_prob_dms_left_Quarter = 0

    # 2. EIC
    # 내림차순 정렬
    df_only_eic_right = df_only_eic.sort_values(by=['Pred_EIC_R'], axis=0, ascending=False)
    df_only_eic_left = df_only_eic.sort_values(by=['Pred_EIC_L'], axis=0, ascending=False)
    # Threshold 이상인 값만 파싱
    series_prob_eic_right = df_only_eic_right[df_only_eic_right['Pred_EIC_R'] > th_EIC_Slice_binary]
    series_prob_eic_left = df_only_eic_left[df_only_eic_left['Pred_EIC_L'] > th_EIC_Slice_binary]
    # 평균값
    if len(series_prob_eic_right) > 0:
        rp_prob_eic_right = sum(series_prob_eic_right['Pred_EIC_R']) / len(series_prob_eic_right)
    else:
        rp_prob_eic_right = 0
    if len(series_prob_eic_left) > 0:
        rp_prob_eic_left = sum(series_prob_eic_left['Pred_EIC_L']) / len(series_prob_eic_left)
    else:
        rp_prob_eic_left = 0

    # 3. EIC / DMS 대표 Probability 조합
    patient_prob_lvo_right_Half = (rp_prob_dms_right_Half * weight_DMS) + (rp_prob_eic_right * weight_EIC)
    patient_prob_lvo_left_Half = (rp_prob_dms_left_Half * weight_DMS) + (rp_prob_eic_left * weight_EIC)
    patient_prob_lvo_right_Quarter = (rp_prob_dms_right_Quarter * weight_DMS) + (rp_prob_eic_right * weight_EIC)
    patient_prob_lvo_left_Quarter = (rp_prob_dms_left_Quarter * weight_DMS) + (rp_prob_eic_left * weight_EIC)

    # 4. LVO 분류 결과 추가
    if patient_prob_lvo_right_Half > th_LVO_Patient_binary:
        patient_binary_lvo_right_Half = 1
    else:
        patient_binary_lvo_right_Half = 0
    if patient_prob_lvo_left_Half > th_LVO_Patient_binary:
        patient_binary_lvo_left_Half = 1
    else:
        patient_binary_lvo_left_Half = 0

    if patient_prob_lvo_right_Quarter > th_LVO_Patient_binary:
        patient_binary_lvo_right_Quarter = 1
    else:
        patient_binary_lvo_right_Quarter = 0
    if patient_prob_lvo_left_Quarter > th_LVO_Patient_binary:
        patient_binary_lvo_left_Quarter = 1
    else:
        patient_binary_lvo_left_Quarter = 0

    # 5. 결과 Dataframe에 환자 결과값 추가
    df_result = df_result.append(
        {'Patient_ID': i,
         'Prob_DMS_Right_Half': rp_prob_dms_right_Half, 'Prob_DMS_Left_Half': rp_prob_dms_left_Half,
         'Prob_DMS_Right_Quarter': rp_prob_dms_right_Quarter, 'Prob_DMS_Left_Quarter': rp_prob_dms_left_Quarter,
         'Prob_EIC_Right': rp_prob_eic_right, 'Prob_EIC_Left': rp_prob_eic_left,
         'Prob_LVO_Right_Half': patient_prob_lvo_right_Half, 'Prob_LVO_Left_Half': patient_prob_lvo_left_Half,
         'Binary_LVO_Right_Half': patient_binary_lvo_right_Half, 'Binary_LVO_Left_Half': patient_binary_lvo_left_Half,
         'Prob_LVO_Right_Quarter': patient_prob_lvo_right_Quarter, 'Prob_LVO_Left_Quarter': patient_prob_lvo_left_Quarter,
         'Binary_LVO_Right_Quarter': patient_binary_lvo_right_Quarter, 'Binary_LVO_Left_Quarter': patient_binary_lvo_left_Quarter
         }, ignore_index=True)


    # 5. 각 DMS / EIC / LVO 별로 Bianry Class 적기

# 컬럼 순서 재배치
df_result = df_result[["Patient_ID",
                       "Prob_DMS_Left_Half", "Prob_DMS_Right_Half",
                       "Prob_DMS_Left_Quarter", "Prob_DMS_Right_Quarter",
                       "Prob_EIC_Left", "Prob_EIC_Right",
                       "Prob_LVO_Left_Half", "Prob_LVO_Right_Half", "Binary_LVO_Right_Half", "Binary_LVO_Left_Half",
                       "Prob_LVO_Left_Quarter", "Prob_LVO_Right_Quarter", "Binary_LVO_Right_Quarter", "Binary_LVO_Left_Quarter"]]

df_result.to_csv('test.csv')
