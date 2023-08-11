# 2023.01.26
# - RSNA와, 김시온 신경외과의 Annotation (nrrd 파일)이 상이한 리스트 뽑기

import pandas as pd
import shutil
import os
from glob import glob
import nrrd

# Nrrd의 segment value별 subtype
dict_subtype = {0: 'BG', 1: 'EDH', 2: 'ICH', 3: 'IVH',
                4: 'SAH', 5: 'SDH', 6: 'SDH(Chronic)', 7: 'HemorrhagicContusion'}
# 원래 RSNA GT와 비교할 필요 없는 subtype 리스트
ignore_subtype = ['BG', 'SDH(Chronic)', 'HemorrhagicContusion']

# RSNA와, Annotation이 다른 리스트 (Series, Annot 양성, RSNA 양성, 서로 다른 Subtype 종류)
list_error_Series = []
list_error_Positive_Annot = []
list_error_Positive_RSNA = []
list_error_Subtype = []

path_label = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Patient_wise_ALL (Subtype 정보 포함).csv'
path_nrrd = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230120_1차 수령 데이터\ALL'
list_nrrd = glob(path_nrrd + '/*.nrrd')

df_label = pd.read_csv(path_label)

for file in list_nrrd:
    # 파일명
    fn = os.path.basename(file)
    # fn_ID = fn.split('-label')[0][5:]
    fn_ID = fn[5:18]

    # 환자의 Subtype 양성여부 확인
    patient_label = df_label[df_label['ID_Series']==fn_ID].reset_index().to_dict()
    dict_gt = {'BG': 0, 'EDH': patient_label['Label_epidural'][0], 'ICH': patient_label['Label_intraparenchymal'][0],
               'IVH': patient_label['Label_intraventricular'][0], 'SAH': patient_label['Label_subarachnoid'][0],
               'SDH': patient_label['Label_subdural'][0], 'SDH(Chronic)': 0, 'HemorrhagicContusion': 0}

    # Nrrd 로드
    data, header = nrrd.read(file)

    # 해당 nrrd의 segment 별 카운트
    segment_count = {'BG': 0, 'EDH': 0, 'ICH': 0, 'IVH': 0, 'SAH': 0, 'SDH': 0, 'SDH(Chronic)': 0,
                     'HemorrhagicContusion': 0} #{}

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                voxel = data[i, j, k]
                name_subtype = dict_subtype.get(voxel)
                if name_subtype not in segment_count:
                    segment_count[name_subtype] = 1
                else:
                    segment_count[name_subtype] += 1
    for key in segment_count:
        if segment_count[key] > 0:
            segment_count[key] = 1

    # Annotation(.nrrd) / GT(RSNA) 의 양성 리스트
    positive_Annot = []
    positive_RSNA = []
    # 서로 다른 key
    diff_keys = []

    for key in segment_count:
        if key == 'BG':
            continue
        if segment_count[key] == 1:
            positive_Annot.append(key)
        if dict_gt[key] == 1:
            positive_RSNA.append(key)

    for key in segment_count:
        # 무시할 subtype 확인
        if key in ignore_subtype:
            continue

        if segment_count[key] != dict_gt[key]:
            diff_keys.append(key)

    if len(diff_keys) > 0:
        print(f'{fn_ID} 환자는 RSNA GT와 다름')
        list_error_Series.append(fn_ID)
        list_error_Positive_Annot.append(positive_Annot)
        list_error_Positive_RSNA.append(positive_RSNA)
        list_error_Subtype.append(diff_keys)

    # 에러 데이터에 대해, 데이터프레임 생성 및 저장
    df_error = pd.DataFrame({'ID_Series': list_error_Series, 'Annot': list_error_Positive_Annot,
                             'GT_RSNA': list_error_Positive_RSNA, 'Diff_Subtype': list_error_Subtype})
    try:
        df_error.to_csv('20230208_RSNA, Annotation 상이한 리스트.csv', index=False)
    except:
        df_error.to_csv('20230208_RSNA, Annotation 상이한 리스트_ver2.csv', index=False)

# 에러 데이터에 대해, 데이터프레임 생성 및 저장
df_error = pd.DataFrame({'ID_Series': list_error_Series, 'Annot': list_error_Positive_Annot,
                         'GT_RSNA': list_error_Positive_RSNA, 'Diff_Subtype': list_error_Subtype})
try:
    df_error.to_csv('20230208_RSNA, Annotation 상이한 리스트.csv', index=False)
except:
    df_error.to_csv('20230208_RSNA, Annotation 상이한 리스트_ver2.csv', index=False)