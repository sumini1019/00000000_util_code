
### 목적 ###
# -> 환자의 슬라이스 별 probability 값을 사용하여,
# -> 환자 별 dms 여부 결정

import pandas as pd
import random
import shutil
import os
#
# # 임계값 - 슬라이스의 dms 여부
# threshold_dms = 0.9
# # 임계값 - 전체 슬라이스 대비, dms 슬라이스의 비율
# threshold_rate_dms = 0.25
# # 임계값 - 전체 슬라이스 대비, dms 슬라이스 개수
# threshold_num_dms = 0
#
# # 환자 번호 없는 리스트
# list_no_patient = []
#
# # 전체 환자에 대한, 성능 카운트
# TP_all = 0
# TN_all = 0
# FP_all = 0
# FN_all = 0
#
# # csv 데이터 프레임 로드
# df_probability = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/DATA_Annotation_list_v210521.csv')
#
# # 환자 별 결과 저장할 데이터 프레임 생성
# # df_result = pd.DataFrame({"Patient_ID", "Prob_DMS_Left", "Prob_DMS_Right", "Prob_EIC_Left", "Prob_EIC_Right", "Prob_LVO_Left", "Prob_LVO_Right"})
# df_result = pd.DataFrame(columns={"Patient_ID", "Num_Total_Slice",
#                                   "GT_Right", "GT_Left",
#                                   "Num_DMS_Slice_Right", "Num_DMS_Slice_Left",
#                                   "Patient_DMS_Right", "Patient_DMS_Left",
#                                   "TP_Right", "TN_Right", "FP_Right", "FN_Right",
#                                   "TP_Left", "TN_Left", "FP_Left", "FN_Left",
#                                   "Patient_Diagnosis"})
#
# # 환자 번호 수
# num_patient = max(df_probability['ID'])
#
# # 환자 번호 리스트
# list_patient = []
# list_patient.append(df_probability['ID'])
# list_patient = list_patient[0]
#
# list_patient = set(list_patient)
# list_patient = list(list_patient)
#
# # df_probability = df_probability[['ID', 'ID_Filename', 'DMS_R', 'DMS_L', 'Pred_DMS_R_Quarter', 'Pred_DMS_L_Quarter']]
#
#
#
# random.shuffle(list_patient)
#
# # 5fold 별 환자 개수
# num_fold = int(len(list_patient)/5)
#
# # fold 환자 리스트
# fold_1 = list_patient[:num_fold]
# fold_2 = list_patient[num_fold:num_fold*2]
# fold_3 = list_patient[num_fold*2:num_fold*3]
# fold_4 = list_patient[num_fold*3:num_fold*4]
# fold_5 = list_patient[num_fold*4:]
#
# # 폴드 별 DMS, OI, EIC 개수
# cnt_DMS = []
# cnt_OI = []
# cnt_EIC = []
#
# # 폴드 카운트 초기화
# cnt_fold_dms = 0
# cnt_fold_eic = 0
# cnt_fold_oi = 0
#
# for i in fold_1:
#     # 현재 환자의 ID에 해당하는 행 파싱
#     df_cur_patient = df_probability[df_probability['ID'] == i]
#     # 환자의 dms, eic, oi 카운트 누적
#     cnt_fold_dms = cnt_fold_dms + sum(df_cur_patient['DMS - R']) + sum(df_cur_patient['DMS - L'])
#     cnt_fold_eic = cnt_fold_eic + sum(df_cur_patient['EIC - R']) + sum(df_cur_patient['EIC - L'])
#     cnt_fold_oi = cnt_fold_oi + sum(df_cur_patient['OI - R']) + sum(df_cur_patient['OI - L'])
#
# # 리스트에 삽입
# cnt_DMS.append(cnt_fold_dms)
# cnt_EIC.append(cnt_fold_eic)
# cnt_OI.append(cnt_fold_oi)
#
# # 폴드 카운트 초기화
# cnt_fold_dms = 0
# cnt_fold_eic = 0
# cnt_fold_oi = 0
#
# for i in fold_2:
#     # 현재 환자의 ID에 해당하는 행 파싱
#     df_cur_patient = df_probability[df_probability['ID'] == i]
#     # 환자의 dms, eic, oi 카운트 누적
#     cnt_fold_dms = cnt_fold_dms + sum(df_cur_patient['DMS - R']) + sum(df_cur_patient['DMS - L'])
#     cnt_fold_eic = cnt_fold_eic + sum(df_cur_patient['EIC - R']) + sum(df_cur_patient['EIC - L'])
#     cnt_fold_oi = cnt_fold_oi + sum(df_cur_patient['OI - R']) + sum(df_cur_patient['OI - L'])
#
# # 리스트에 삽입
# cnt_DMS.append(cnt_fold_dms)
# cnt_EIC.append(cnt_fold_eic)
# cnt_OI.append(cnt_fold_oi)
#
# # 폴드 카운트 초기화
# cnt_fold_dms = 0
# cnt_fold_eic = 0
# cnt_fold_oi = 0
#
# for i in fold_3:
#     # 현재 환자의 ID에 해당하는 행 파싱
#     df_cur_patient = df_probability[df_probability['ID'] == i]
#     # 환자의 dms, eic, oi 카운트 누적
#     cnt_fold_dms = cnt_fold_dms + sum(df_cur_patient['DMS - R']) + sum(df_cur_patient['DMS - L'])
#     cnt_fold_eic = cnt_fold_eic + sum(df_cur_patient['EIC - R']) + sum(df_cur_patient['EIC - L'])
#     cnt_fold_oi = cnt_fold_oi + sum(df_cur_patient['OI - R']) + sum(df_cur_patient['OI - L'])
#
# # 리스트에 삽입
# cnt_DMS.append(cnt_fold_dms)
# cnt_EIC.append(cnt_fold_eic)
# cnt_OI.append(cnt_fold_oi)
#
# # 폴드 카운트 초기화
# cnt_fold_dms = 0
# cnt_fold_eic = 0
# cnt_fold_oi = 0
#
# for i in fold_4:
#     # 현재 환자의 ID에 해당하는 행 파싱
#     df_cur_patient = df_probability[df_probability['ID'] == i]
#     # 환자의 dms, eic, oi 카운트 누적
#     cnt_fold_dms = cnt_fold_dms + sum(df_cur_patient['DMS - R']) + sum(df_cur_patient['DMS - L'])
#     cnt_fold_eic = cnt_fold_eic + sum(df_cur_patient['EIC - R']) + sum(df_cur_patient['EIC - L'])
#     cnt_fold_oi = cnt_fold_oi + sum(df_cur_patient['OI - R']) + sum(df_cur_patient['OI - L'])
#
# # 리스트에 삽입
# cnt_DMS.append(cnt_fold_dms)
# cnt_EIC.append(cnt_fold_eic)
# cnt_OI.append(cnt_fold_oi)
#
# # 폴드 카운트 초기화
# cnt_fold_dms = 0
# cnt_fold_eic = 0
# cnt_fold_oi = 0
#
# for i in fold_5:
#     # 현재 환자의 ID에 해당하는 행 파싱
#     df_cur_patient = df_probability[df_probability['ID'] == i]
#     # 환자의 dms, eic, oi 카운트 누적
#     cnt_fold_dms = cnt_fold_dms + sum(df_cur_patient['DMS - R']) + sum(df_cur_patient['DMS - L'])
#     cnt_fold_eic = cnt_fold_eic + sum(df_cur_patient['EIC - R']) + sum(df_cur_patient['EIC - L'])
#     cnt_fold_oi = cnt_fold_oi + sum(df_cur_patient['OI - R']) + sum(df_cur_patient['OI - L'])
#
# # 리스트에 삽입
# cnt_DMS.append(cnt_fold_dms)
# cnt_EIC.append(cnt_fold_eic)
# cnt_OI.append(cnt_fold_oi)
#
# print("DMS : ", cnt_DMS)
# print("EIC : ", cnt_EIC)
# print("OI : ", cnt_OI)
#
# df_fold_1 = pd.DataFrame()
# df_fold_2 = pd.DataFrame()
# df_fold_3 = pd.DataFrame()
# df_fold_4 = pd.DataFrame()
# df_fold_5 = pd.DataFrame()
#
# for i in fold_1:
#     df_fold_1 = df_fold_1.append(df_probability[df_probability['ID'] == i])
# for i in fold_2:
#     df_fold_2 = df_fold_2.append(df_probability[df_probability['ID'] == i])
# for i in fold_3:
#     df_fold_3 = df_fold_3.append(df_probability[df_probability['ID'] == i])
# for i in fold_4:
#     df_fold_4 = df_fold_4.append(df_probability[df_probability['ID'] == i])
# for i in fold_5:
#     df_fold_5 = df_fold_5.append(df_probability[df_probability['ID'] == i])
#
# df_fold_1.to_csv('df_fold_1.csv')
# df_fold_2.to_csv('df_fold_2.csv')
# df_fold_3.to_csv('df_fold_3.csv')
# df_fold_4.to_csv('df_fold_4.csv')
# df_fold_5.to_csv('df_fold_5.csv')
#
#


################### DMS 파일들만, fold별 복사

# 복사할 경로
resultPath_dms = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/dms'
resultPath_normal = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/normal'
# resultPath_dms = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/dms(for_test)'
# resultPath_normal = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/normal(for_test)'

# 복사할 원본 파일 경로
path = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold_split/4'

# csv 데이터 프레임 로드
df = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/df_fold_split4.csv')
df = df[df['USED for DMS'] == 1]

# 인덱스 초기화
df.reset_index(drop=True, inplace=True)

# csv 파일 내, 행 개수만큼 반복
for i in range(0, len(df)):

    # 방향 상관없는 파일 이름 파싱
    name_file = df.loc[i].ID_Filename

    # Left / Right 파일 네이밍
    name_file_left = name_file.replace('.dcm', '_L.dcm')
    name_file_right = name_file.replace('.dcm', '_R.dcm')

    # Left가 DMS 라면, 파일 이동
    if df.loc[i]["DMS - L"] == 1:
        try:
            shutil.copy(os.path.join(path, name_file_left), os.path.join(resultPath_dms, name_file_left))
            # print("Succesee Copy : {}".format(name_file_left))
        except:
            print("False Copy : {}".format(name_file_left))
    # DMS가 아니라면, normal에 copy
    else:
        try:
            shutil.copy(os.path.join(path, name_file_left), os.path.join(resultPath_normal, name_file_left))
            # print("Succesee Copy : {}".format(name_file_left))
        except:
            print("False Copy : {}".format(name_file_left))

    # Right도 똑같이 적용
    if df.loc[i]["DMS - R"] == 1:
        try:
            shutil.copy(os.path.join(path, name_file_right), os.path.join(resultPath_dms, name_file_right))
            # print("Succesee Copy : {}".format(name_file_right))
        except:
            print("False Copy : {}".format(name_file_right))
    # DMS가 아니라면, normal에 copy
    else:
        try:
            shutil.copy(os.path.join(path, name_file_right), os.path.join(resultPath_normal, name_file_right))
            # print("Succesee Copy : {}".format(name_file_right))
        except:
            print("False Copy : {}".format(name_file_right))