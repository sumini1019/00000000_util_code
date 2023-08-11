# 2023.01.27
# - csv 읽어서, 해당 리스트에 해당하는
# - RSNA Annotation, 탑병원 Annotation 파일 읽기

import pandas as pd
import shutil
import os
from glob import glob
import nrrd

path_label = r'D:\OneDrive\00000000_Code\00000000_util_code\20230208_RSNA, Annotation 상이한 리스트.csv'
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
    for item in list_nifti:
        if cur_ID in item:
            ID_include_Index = os.path.basename(item)

            # 데이터프레임의 값 변경
            df_label.loc[index, 'ID_Series'] = ID_include_Index

    df_label.to_csv('20230208_RSNA, Annotation 상이한 리스트.csv', index=False)