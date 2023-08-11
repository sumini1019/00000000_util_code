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

pd.set_option('display.precision', 4)

# path_src_csv = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS'
# list_inst = ['1_AJUMC', '2_GMC', '3_EUMC', '4_CSUH', '5_CNUSH', '6_SCHMC', '7_etc']
#
# result_list = []
#
# for inst in list_inst:
#     path_inst_root = os.path.join(path_src_csv, inst, 'F_Indv_Annotation')
#     # csv 리스트
#     list_inst_csv = os.listdir(path_inst_root)
#
#     # csv마다 순회하면서, 정보 읽기
#     for patient_csv in list_inst_csv:
#         # 결과 초기값
#         dict_result = {'HR-ID': patient_csv[:9],
#                        'EIC_L': -1, 'EIC_R': -1,
#                        'DMS_L': -1, 'DMS_R': -1,
#                        'OI_L': -1, 'OI_R': -1}
#
#         list_EDO = ['EIC_L', 'EIC_R', 'DMS_L', 'DMS_R', 'OI_L', 'OI_R']
#
#         df_patient = pd.read_csv(os.path.join(path_inst_root, patient_csv), skiprows=[0])
#         df_patient = df_patient.drop(['Slice #', '-'], axis=1)
#         df_patient = df_patient.rename(
#             columns={'R-DMS': 'DMS_R', 'R-EIC': 'EIC_R', 'R-OI': 'OI_R', 'L-DMS': 'DMS_L', 'L-EIC': 'EIC_L',
#                      'L-OI': 'OI_L'})
#
#         # 각 열의 합
#         sum_df = df_patient.sum()
#
#         # 질환 / 방향 별 결과 저장
#         for item in list_EDO:
#             if sum_df[f'{item}'] > 0:
#                 dict_result[f'{item}'] = 1
#             elif sum_df[f'{item}'] == 0:
#                 dict_result[f'{item}'] = 0
#             else:
#                 print(f'ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRR - {patient_csv}')
#
#         # 결과 리스트에 추가
#         result_list.append(dict_result)
#
#     # 결과 리스트를 데이터프레임으로 변환
#     df_result = pd.DataFrame(result_list)
#     # csv 파일로 저장
#     try:
#         df_result.to_csv('20230222_FinalListv3_미완성.csv', index=False)
#     except:
#         df_result.to_csv('20230222_FinalListv3_미완성_sub.csv', index=False)
#
# # 결과 리스트를 데이터프레임으로 변환
# df_result = pd.DataFrame(result_list)
# # csv 파일로 저장
# try:
#     df_result.to_csv('20230222_FinalListv3_미완성.csv', index=False)
# except:
#     df_result.to_csv('20230222_FinalListv3_미완성_sub.csv', index=False)




path_v2 = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\Final_List_v2.csv'
path_v3 = r'D:\OneDrive\00000000_Code\00000000_util_code\20230222_FinalListv3_미완성.csv'

# v2의 HR-ID  /  v3의 HR-ID 를 기준으로 merge
df_v2 = pd.read_csv(path_v2)
df_v3 = pd.read_csv(path_v3)

merged_df = pd.merge(df_v2, df_v3, on='HR-ID')

merged_df.to_csv('20230222_FinalList_v3_EDO_반구결과포함.csv')