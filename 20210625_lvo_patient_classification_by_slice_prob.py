### 목적 ###
# -> 환자의 슬라이스 별 DMS, EIC probability 값을 사용하여,
# -> 환자 별 lvo 여부 결정

import pandas as pd
import os

# 임계값 - 슬라이스의 dms 여부
threshold_lvo = 0.6
# 임계값 - 전체 슬라이스 대비, lvo 슬라이스 개수 (임계값 이상 시, lvo)
threshold_num_lvo = 2
# DMS / EIC Probability 조합할 weight
slice_weight_dms = 0.7
slice_weight_eic = 0.3

# 환자 번호 없는 리스트
list_no_patient = []

# 전체 환자에 대한, 성능 카운트
TP_all = 0
TN_all = 0
FP_all = 0
FN_all = 0

# 결과 데이터프레임
df_result = pd.DataFrame(columns={"Patient_ID", "Num_Total_Slice",
                                  "GT_Right", "GT_Left",
                                  "Num_LVO_Slice_Right", "Num_LVO_Slice_Left",
                                  "Patient_LVO_Right", "Patient_LVO_Left",
                                  "TP_Right", "TN_Right", "FP_Right", "FN_Right",
                                  "TP_Left", "TN_Left", "FP_Left", "FN_Left",
                                  "Patient_Diagnosis"})

# probability 데이터
path_csv = 'Z:/Sumin_Jung/00000000_RESULT/2_cELVO/20210406_Vessel에 대한 DMS Classification 결과 정리/11. EIC, DMS embedding 결합 통한 LVO 예측/1. Slice 별 결과 조합'
df_probability = pd.read_csv(os.path.join(path_csv, '(Data)20210625_EIC_DMS_Slice별 Probability.csv'))

# lvo GT 데이터
path_label_lvo = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/LVO_GT_v210602_re_modify_bySumin.csv'
df_label_lvo = pd.read_csv(path_label_lvo)

# GT가 존재하는 환자번호 리스트
list_lvo_patient = list(set(df_label_lvo['ID']))

df_probability['CTA_R'] = -1
df_probability['CTA_L'] = -1

# CTA LVO 레이블 저장
for i in range(0, len(df_probability)):
    cur_id = df_probability.loc[i].ID
    if cur_id in list_lvo_patient:
        df_probability['CTA_R'][i] = df_label_lvo[df_label_lvo['ID']==cur_id]['CTA_R'].values
        df_probability['CTA_L'][i] = df_label_lvo[df_label_lvo['ID'] == cur_id]['CTA_L'].values

# LVO GT가 존재하지 않는, Probability 행은 삭제
# a. 인덱스 뽑기
idx_no_exist_lvo_trn = df_probability[df_probability['CTA_R'] == -1].index
# b. Trndf에서, 삭제
df_probability = df_probability.drop(idx_no_exist_lvo_trn)
df_probability.reset_index(drop=True, inplace=True)

df_probability['Prob_LVO_R'] = -1.0
df_probability['Prob_LVO_L'] = -1.0

# LVO Probability 계산 (가중치 사용)
for i in range(0, len(df_probability)):
    # DMS / EIC 같이 존재
    if df_probability.iloc[i]['USED for DMS'] == 1:
        # print('R', (df_probability.iloc[i]['Prob_EIC_R'] * slice_weight_eic)\
        #                                + (df_probability.iloc[i]['Prob_DMS_R'] * slice_weight_dms)
        #       )
        df_probability['Prob_LVO_R'][i] = (float)(df_probability.iloc[i]['Prob_EIC_R'] * slice_weight_eic)\
                                       + (df_probability.iloc[i]['Prob_DMS_R'] * slice_weight_dms)
        df_probability['Prob_LVO_L'][i] = (float)(df_probability.iloc[i]['Prob_EIC_L'] * slice_weight_eic)\
                                       + (df_probability.iloc[i]['Prob_DMS_L'] * slice_weight_dms)
    # EIC만 존재
    elif df_probability.iloc[i]['USED for DMS'] == 0:
        df_probability['Prob_LVO_R'][i] = df_probability.iloc[i]['Prob_EIC_R']
        df_probability['Prob_LVO_L'][i] = df_probability.iloc[i]['Prob_EIC_L']
    else:
        print('error : not exist "used_for_dms"')

# 환자 번호 수
num_patient = max(df_probability['ID'])

# df_probability = df_probability[['ID', 'ID_Filename', 'DMS - R', 'DMS - L', 'Prob_DMS_R', 'Prob_DMS_L']]

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
    if sum(df_cur_patient['CTA_R']) > 0:  # right
        label_patient_lvo_right = True
    else:
        label_patient_lvo_right = False
    if sum(df_cur_patient['CTA_L']) > 0:  # left
        label_patient_lvo_left = True
    else:
        label_patient_lvo_left = False

    ###############################################################################
    # 아래는 모두 Result 관련
    ###############################################################################

    # Probability가 threshold 넘는 slice의 개수 계산
    num_slice_lvo_right = list(df_cur_patient['Prob_LVO_R'] > threshold_lvo).count(True)
    num_slice_lvo_left = list(df_cur_patient['Prob_LVO_L'] > threshold_lvo).count(True)

    if num_slice_lvo_right >= threshold_num_lvo:
        patient_lvo_right = True
    else:
        patient_lvo_right = False
    if num_slice_lvo_left >= threshold_num_lvo:
        patient_lvo_left = True
    else:
        patient_lvo_left = False

    # 환자의 Diagnosis 결정
    # - Normal / Left / Right / Bileteral
    if patient_lvo_right==True and patient_lvo_left==True:
        patient_diagnosis = "Bileteral"
    elif patient_lvo_right==True and patient_lvo_left==False:
        patient_diagnosis = "Right"
    elif patient_lvo_right == False and patient_lvo_left == True:
        patient_diagnosis = "Left"
    else:
        patient_diagnosis = "Normal"

    # 정답과, 모델 결과의 일치 판정
    if label_patient_lvo_right==True and patient_lvo_right==True:
        TP_right = 1
    elif label_patient_lvo_right==True and patient_lvo_right==False:
        FN_right = 1
    elif label_patient_lvo_right == False and patient_lvo_right == True:
        FP_right = 1
    else:
        TN_right = 1

    if label_patient_lvo_left==True and patient_lvo_left==True:
        TP_left = 1
    elif label_patient_lvo_left==True and patient_lvo_left==False:
        FN_left = 1
    elif label_patient_lvo_left == False and patient_lvo_left == True:
        FP_left = 1
    else:
        TN_left = 1

    # 현재 환자 결과 저장
    df_result = df_result.append(
        {"Patient_ID": i, "Num_Total_Slice": len(df_cur_patient),
         "GT_Right":label_patient_lvo_right, "GT_Left":label_patient_lvo_left,
         "Num_LVO_Slice_Right": num_slice_lvo_right, "Num_LVO_Slice_Left": num_slice_lvo_left,
         "Patient_LVO_Right": patient_lvo_right, "Patient_LVO_Left": patient_lvo_left,
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
                       "Num_LVO_Slice_Right", "Num_LVO_Slice_Left",
                       "Patient_LVO_Right", "Patient_LVO_Left",
                       "TP_Right", "TN_Right", "FP_Right", "FN_Right",
                       "TP_Left", "TN_Left", "FP_Left", "FN_Left",
                       "Patient_Diagnosis"]]

# 최종 결과 저장
df_result.to_csv(os.path.join(path_csv, '(Result)DMS_EIC_슬라이스_Prob조합.csv'))

# 전체 환자에 대한, 지표 계산
result_num_case = TP_all + TN_all + FP_all + FN_all
result_accuracy = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
result_precision = TP_all / (TP_all + FP_all)
result_sensitivity = TP_all / (TP_all + FN_all)
result_specificity = TN_all / (TN_all + FP_all)

print("Num Case : ", result_num_case)
print("TP : ", TP_all)
print("TN : ", TN_all)
print("FP : ", FP_all)
print("FN : ", FN_all)
print("Accuracy : ", result_accuracy)
print("Precision : ", result_precision)
print("Sensitivity : ", result_sensitivity)
print("Specificity : ", result_specificity)