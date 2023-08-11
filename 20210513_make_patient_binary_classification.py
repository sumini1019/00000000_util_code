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

# 환자 최종 LVO Probability에서, LVO 여부 결정할 Probability 임계값
th_rate_slice_LVO = 0.1

# 환자 번호 없는 리스트
list_no_patient = []

# csv 데이터 프레임 로드
df_probability = pd.read_csv('Z:/Sumin_Jung/00000000_RESULT/2_cELVO/20210406_Vessel에 대한 DMS Classification 결과 정리/9. EIC 결합 반구 선택 데이터 사용/Probability_DMS_EIC_binary_classification.csv')

# 환자 별 결과 저장할 데이터 프레임 생성
# df_result = pd.DataFrame({"Patient_ID", "Prob_DMS_Left", "Prob_DMS_Right", "Prob_EIC_Left", "Prob_EIC_Right", "Prob_LVO_Left", "Prob_LVO_Right"})
df_result = pd.DataFrame(columns={"Patient_ID", "Num_Total_Slice",
                       "Num_LVO_Right_Half", "Num_LVO_Left_Half",
                       "Num_LVO_Right_Quarter", "Num_LVO_Left_Quarter",
                       "Binary_LVO_Right_Half", "Binary_LVO_Left_Half",
                       "Binary_LVO_Right_Quarter", "Binary_LVO_Left_Quarter"})

# 환자 번호 수
num_patient = max(df_probability['ID'])

df_probability = df_probability[['ID', 'ID_Filename', 'Bi_LVO_L_Half', 'Bi_LVO_R_Half', 'Bi_LVO_L_Quarter', 'Bi_LVO_R_Quarter']]

# 환자 번호 순으로 반복
for i in range(1, num_patient+1):
    # 현재 환자의 데이터 파싱
    df_cur_patient = df_probability[df_probability['ID'] == i]

    # 해당 환자 번호의 데이터가 없을 경우 다음 번호로 넘어갈 것
    if len(df_cur_patient) == 0:
        list_no_patient.append(i)
        continue

    # Slice 개수
    num_slice_LVO_R_Half = sum(df_cur_patient['Bi_LVO_R_Half'])
    num_slice_LVO_L_Half = sum(df_cur_patient['Bi_LVO_L_Half'])
    num_slice_LVO_R_Quarter = sum(df_cur_patient['Bi_LVO_R_Quarter'])
    num_slice_LVO_L_Quarter = sum(df_cur_patient['Bi_LVO_L_Quarter'])

    # LVO Slice의 비율
    rate_slice_LVO_R_Half = num_slice_LVO_R_Half / len(df_cur_patient)
    rate_slice_LVO_L_Half = num_slice_LVO_L_Half / len(df_cur_patient)
    rate_slice_LVO_R_Quarter = num_slice_LVO_R_Quarter / len(df_cur_patient)
    rate_slice_LVO_L_Quarter = num_slice_LVO_L_Quarter / len(df_cur_patient)

    # thresholding
    # 4. LVO 분류 결과 추가
    if rate_slice_LVO_R_Half > th_rate_slice_LVO:
        patient_binary_lvo_right_Half = 1
    else:
        patient_binary_lvo_right_Half = 0
    if rate_slice_LVO_L_Half > th_rate_slice_LVO:
        patient_binary_lvo_left_Half = 1
    else:
        patient_binary_lvo_left_Half = 0

    if rate_slice_LVO_R_Quarter > th_rate_slice_LVO:
        patient_binary_lvo_right_Quarter = 1
    else:
        patient_binary_lvo_right_Quarter = 0
    if rate_slice_LVO_L_Quarter > th_rate_slice_LVO:
        patient_binary_lvo_left_Quarter = 1
    else:
        patient_binary_lvo_left_Quarter = 0

    # 5. 결과 Dataframe에 환자 결과값 추가
    df_result = df_result.append(
        {'Patient_ID': i, "Num_Total_Slice": len(df_cur_patient),
         'Num_LVO_Right_Half': num_slice_LVO_R_Half, 'Num_LVO_Left_Half': num_slice_LVO_L_Half,
         'Num_LVO_Right_Quarter': num_slice_LVO_R_Quarter, 'Num_LVO_Left_Quarter': num_slice_LVO_L_Quarter,
         'Binary_LVO_Right_Half': patient_binary_lvo_right_Half, 'Binary_LVO_Left_Half': patient_binary_lvo_left_Half,
         'Binary_LVO_Right_Quarter': patient_binary_lvo_right_Quarter, 'Binary_LVO_Left_Quarter': patient_binary_lvo_left_Quarter
         }, ignore_index=True)


    # 5. 각 DMS / EIC / LVO 별로 Bianry Class 적기

# 컬럼 순서 재배치
df_result = df_result[["Patient_ID", "Num_Total_Slice",
                       "Num_LVO_Right_Half", "Num_LVO_Left_Half",
                       "Num_LVO_Right_Quarter", "Num_LVO_Left_Quarter",
                       "Binary_LVO_Right_Half", "Binary_LVO_Left_Half",
                       "Binary_LVO_Right_Quarter", "Binary_LVO_Left_Quarter"]]

df_result.to_csv('test3.csv')
