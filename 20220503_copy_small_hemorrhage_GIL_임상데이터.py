### 목적 ###
# -> 길병원 데이터에서, Small Hemorrhage Annoation 데이터 기반으로 Small hemorrhage 데이터 복사 (Slice, Series)

import pandas as pd
import shutil
import os

# Series ID 데이터 뽑아올 root 폴더
root_data_SeriesID = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210628_cHS_길병원 임상데이터'

# 복사할 위치 (Series)
root_for_copy_series = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 길병원\0. all_series'
# 복사할 위치 (Slice)
path_copy_slice = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 길병원\0. all_slice_dcm'

# Small Hemorrhage 정보 csv
df_merge = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 길병원\20220503_Small Hemorrhage Case_의장님 Annotation.csv')

#########################
# 2. Slice 복사
#########################

# ID Series, index 사용하여 복사
for i in range(0, len(df_merge)):
    # 현재 row의, ID Series / index 파싱
    cur_series = df_merge.loc[i, 'ID_Series']
    cur_index = df_merge.loc[i, 'index']
    # 현재 시리즈 폴더의, 파일 리스트
    cur_path = os.path.join(root_data_SeriesID, cur_series)
    list_cur_data = os.listdir(cur_path)
    # 파일명, path 설정
    target_file_name = list_cur_data[cur_index]
    target_path = os.path.join(cur_path, target_file_name)

    # 복사 시, 폴더명까지 붙여서 이름 변경 후, 복사
    new_target_file_name = cur_series + '_' + target_file_name
    shutil.copyfile(target_path, os.path.join(path_copy_slice, new_target_file_name))

#########################
# 1. Series 복사
#########################

# SeriesID List 뽑기
list_SeriesID = df_merge['ID_Series']

# 리스트 중복제거
my_set = set(list_SeriesID) #집합set으로 변환
list_SeriesID = list(my_set) #list로 변환

cnt_suc = 0
cnt_fail = 0

# SeriesID에 해당하는 폴더 복사
for cur_SeriesID in list_SeriesID:
    try:
        shutil.copytree(os.path.join(root_data_SeriesID, cur_SeriesID), os.path.join(root_for_copy_series, cur_SeriesID))
        print('copy success : {}'.format(cur_SeriesID))
        cnt_suc = cnt_suc + 1
    except:
        print('copy fail : {}'.format(cur_SeriesID))
        cnt_fail = cnt_fail + 1

print('Num Series : {}'.format(cnt_suc + cnt_fail))
print('Num Success : {}'.format(cnt_suc))
print('Num Fail : {}'.format(cnt_fail))

