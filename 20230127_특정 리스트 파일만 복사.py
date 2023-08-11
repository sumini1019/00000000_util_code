# 2023.01.27
# - csv 읽어서, 해당 리스트에 해당하는
# - RSNA Annotation, 탑병원 Annotation 파일 읽기

import pandas as pd
import shutil
import os
from glob import glob
import nrrd

path_label = r'D:\OneDrive\00000000_Code\00000000_util_code\20230126_RSNA, Annotation 상이한 리스트.csv'
path_nifti = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series(ID index 추가)\Hemo'
path_nrrd = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230120_1차 수령 데이터\ALL'
path_dest = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230127_1차 수령 데이터에 대한 검토 대상 데이터'

list_nifti = glob(path_nifti + '/*.nii.gz')
list_nrrd = glob(path_nrrd + '/*.nrrd')

df_label = pd.read_csv(path_label)

for index, row in df_label.iterrows():
    cur_ID = row['ID_Series']

    # ID에 해당하는 Nifti, NRRD 경로 확인
    fn_nifti = ''
    fn_nrrd = ''

    # 1. Nifti
    cnt = 0
    for item in list_nifti:
        if cur_ID in item:
            cnt = cnt + 1
            # print(f'해당하는 nifti 개수 {cnt}')
            fn_nifti = item
    # 복사
    if cnt == 1:
        shutil.copy(fn_nifti, os.path.join(path_dest, os.path.basename(fn_nifti)))
    else:
        print(f'{cur_ID}는 nifti가 없어요')
        continue

    # 2. Nrrd
    cnt = 0
    for item in list_nrrd:
        if cur_ID in item:
            cnt = cnt + 1
            # print(f'해당하는 nifti 개수 {cnt}')
            fn_nrrd = item

    # 복사
    if cnt == 1:
        shutil.copy(fn_nrrd, os.path.join(path_dest, os.path.basename(fn_nrrd)))
    else:
        print(f'{cur_ID}는 nrrd가 없어요 / cnt : {cnt}')
        continue




# for file in list_nrrd:
#     # 파일명
#     fn = os.path.basename(file)
#     # fn_ID = fn.split('-label')[0][5:]
#     fn_ID = fn[5:18]
#
#     # 환자의 Subtype 양성여부 확인
#     patient_label = df_label[df_label['ID_Series']==fn_ID].reset_index().to_dict()
#     dict_gt = {'BG': 0, 'EDH': patient_label['Label_epidural'][0], 'ICH': patient_label['Label_intraparenchymal'][0],
#                'IVH': patient_label['Label_intraventricular'][0], 'SAH': patient_label['Label_subarachnoid'][0],
#                'SDH': patient_label['Label_subdural'][0], 'SDH(Chronic)': 0, 'HemorrhagicContusion': 0}
#
#     # Nrrd 로드
#     data, header = nrrd.read(file)
#
#     # 해당 nrrd의 segment 별 카운트
#     segment_count = {'BG': 0, 'EDH': 0, 'ICH': 0, 'IVH': 0, 'SAH': 0, 'SDH': 0, 'SDH(Chronic)': 0,
#                      'HemorrhagicContusion': 0} #{}
#
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             for k in range(data.shape[2]):
#                 voxel = data[i, j, k]
#                 name_subtype = dict_subtype.get(voxel)
#                 if name_subtype not in segment_count:
#                     segment_count[name_subtype] = 1
#                 else:
#                     segment_count[name_subtype] += 1
#     for key in segment_count:
#         if segment_count[key] > 0:
#             segment_count[key] = 1
#
#     # 서로 다른 key를 찾자
#     diff_keys = []
#
#     for key in segment_count:
#         # 무시할 subtype 확인
#         if key in ignore_subtype:
#             continue
#
#         if segment_count[key] != dict_gt[key]:
#             diff_keys.append(key)
#
#     if len(diff_keys) > 0:
#         print(f'{fn_ID} 환자는 RSNA GT와 다름')
#         list_Series_error.append(fn_ID)
#         list_diff_subtype.append(diff_keys)
#
#     # 에러 데이터에 대해, 데이터프레임 생성 및 저장
#     df_error = pd.DataFrame({'ID_Series': list_Series_error, 'Diff_Subtype': list_diff_subtype})
#     try:
#         df_error.to_csv('20230126_RSNA, Annotation 상이한 리스트.csv', index=False)
#     except:
#         df_error.to_csv('20230126_RSNA, Annotation 상이한 리스트_ver2.csv', index=False)
#
# # 에러 데이터에 대해, 데이터프레임 생성 및 저장
# df_error = pd.DataFrame({'ID_Series': list_Series_error, 'Diff_Subtype': list_diff_subtype})
# try:
#     df_error.to_csv('20230126_RSNA, Annotation 상이한 리스트.csv', index=False)
# except:
#     df_error.to_csv('20230126_RSNA, Annotation 상이한 리스트_ver2.csv', index=False)