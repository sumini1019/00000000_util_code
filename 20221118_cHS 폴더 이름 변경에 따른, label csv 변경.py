### 목적 ###
# -> cHS 폴더명 변경 후,
# -> 폴더명 변경 한 데이터에 대해, csv label명 변경

import pandas as pd
import random
import shutil
import os

# 이름 변경된 폴더 경로
path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\1. 2D Annotation (일부)\2. Data_Series (png, label)\Hemo'

# 라벨 (슬라이스, 시리즈)
df_label_slice = pd.read_csv(r'C:\Users\user\Downloads\cHS_Label_Slice_wise.csv')
df_label_series = pd.read_csv(r'C:\Users\user\Downloads\cHS_Label_Patient_wise.csv')

# 폴더 리스트
list_dir = os.listdir(path_root)

# Dataframe의 Patient ID 수정
for cur_dir in list_dir:
    # 폴더 내 idx 제외한 Series ID만 파싱
    cur_series_ID = cur_dir[5:]

    # 라벨 DF 에서, 해당하는 행 찾고, 수정
    try:
        # Series에서, 해당하는 행의 Patient ID 수정
        df_label_series.loc[df_label_series['ID_Series']==cur_series_ID, 'ID_Series'] = cur_dir
        # Slice에서, 해당하는 행의 Patient ID 수정
        df_label_slice.loc[df_label_slice['ID_Series'] == cur_series_ID, 'ID_Series'] = cur_dir
    except:
        print('실패 - {}'.format(cur_dir))
        continue

# Dataframe 정렬 후 저장
# - 저장한 후에, index 없는건 모조리 삭제 (Normal 이므로)
# - 코드 찾기 귀찮다..
df_label_series = df_label_series.sort_values('ID_Series')
df_label_slice = df_label_slice.sort_values('ID_Series')
df_label_series.to_csv('cHS_Label_Patient_wise_sorted.csv', index=False)
df_label_slice.to_csv('cHS_Label_Slice_wise_sorted.csv', index=False)