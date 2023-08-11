# 2022.04.22
# - seriesID 에 해당하는, 데이터를 뽑아오는 코드

import pandas as pd
import shutil
import os

# Series ID 데이터 뽑아올 root 폴더
root_data_SeriesID = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID★★★'

# 데이터 복사할 타겟 경로
# root_for_copy = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage (new)\0. all_series (new)'
# root_for_copy = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage (new)\0. all_series (non_small)'
root_for_copy = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_series (new)'


# Small Hemorrhage 정보 csv
# df_merge = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\1_cHS\20220422_cHS Small Hemorrhage 논문용 성능 정리\result_Non_Small_Hemorrhage_merged.csv')
df_merge = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\Label_Slice_Wise (Small Hemorrhage 2차_검토 후).csv')

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
        shutil.copytree(os.path.join(root_data_SeriesID, cur_SeriesID), os.path.join(root_for_copy, cur_SeriesID))
        print('copy success : {}'.format(cur_SeriesID))
        cnt_suc = cnt_suc + 1
    except:
        print('copy fail : {}'.format(cur_SeriesID))
        cnt_fail = cnt_fail + 1

print('Num Series : {}'.format(cnt_suc + cnt_fail))
print('Num Success : {}'.format(cnt_suc))
print('Num Fail : {}'.format(cnt_fail))