### 목적 ###
# -> dicom 데이터의 meta data 읽어오고
# -> meta data 중, Series ID 읽어와서, 새로운 DataFrame 만들기

import pandas as pd
import random
import shutil
import os
from glob import glob
import pydicom

path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID'

# Series 폴더 리스트
list_series = os.listdir(path_root)

# list_series = list_series[:2]

# Hemo Label 정보
df_label = pd.read_csv(r'D:\00000000_Data\hemorrhage_classification\rsna_train_binary.csv')

# 카운트
cnt = 0

# 각 폴더마다, 순회
for cur_series in list_series:
    # 폴더의 Slice 리스트 읽어오기
    list_slice = sorted(os.listdir(os.path.join(path_root, cur_series)))
    list_slice = [file for file in list_slice if file.endswith(".dcm")]

    # 문자열 중 '.dcm' 제거
    for i in range(len(list_slice)):
        if '.dcm' in list_slice[i]:
            list_slice[i] = list_slice[i].replace('.dcm', '')

    # Series의 DataFrame 생성
    df_cur_series = pd.DataFrame(list_slice, columns=['Image'])
    df_cur_series['ID_Series'] = cur_series

    # 합계 DataFrame 복사 or append
    if cnt == 0:
        df_all = df_cur_series.copy()
    else:
        df_all = pd.concat([df_all, df_cur_series], ignore_index=True)

    # 카운트 증가
    cnt = cnt+1

    # csv 임시저장
    df_all.to_csv('df_all_temp.csv')

    if cnt % 10000 == 0:
        # Slice의 Hemo 정보 추가 (by DataFrame Join 함수 사용)
        df_label_copy = df_label.set_index('Image')
        df_all_copy = df_all.set_index('Image')
        df_join = df_all_copy.join(df_label_copy).reset_index()

        # 최종 결과 저장
        df_join.to_csv('result_cHS_Label.csv')

# Slice의 Hemo 정보 추가 (by DataFrame Join 함수 사용)
df_label.set_index('Image', inplace=True)
df_all.set_index('Image', inplace=True)
df_join = df_all.join(df_label).reset_index()

# 최종 결과 저장
df_join.to_csv('result_cHS_Label.csv')

