import pandas as pd
import random
import shutil
import os

path_csv = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\3. Ground Truth (Index 추가)\cHS_Label_Patient_wise_indexed (only_Normal).csv'
path_src = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series\Normal_temp'
path_dst = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series\Normal_temp_modified'

os.makedirs(path_dst, exist_ok=True)

df = pd.read_csv(path_csv)

list_nifti = os.listdir(path_src)

for idx, fn_nifti in enumerate(list_nifti):
    fn_nifti_without_ext = fn_nifti.replace('.nii.gz', '')

    fn_nifti_with_index = df[df['ID_Series'].str.contains(fn_nifti_without_ext)]['ID_Series'].values[0]

    fn_new = fn_nifti_with_index + '.nii.gz'

    path_cur_src = os.path.join(path_src, fn_nifti)
    path_cur_dst = os.path.join(path_dst, fn_new)

    shutil.copy(path_cur_src, path_cur_dst)