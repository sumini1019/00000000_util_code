# 2022.06.16
# - Slice가 모여있는 폴더에서
# - 각 슬라이스의 SeriesID 읽고, 해당하는 SeriesID만 복사해오는 코드

import pandas as pd
import shutil
import os
from glob import glob

# Series ID 확인 할, Slice가 모여있는 폴더
path_target = r'H:\20220614_이어명 인턴 전달\1. cHS_Annotation_DataSet\2. Fold별 DataSet\fold_10'
# Series ID 데이터를 복사해 올 폴더 (SeiresID 별로 폴더 별개)
path_source = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID★★★'
# 복사 할 경로
path_dest = r'H:\20220614_이어명 인턴 전달\1. cHS_Annotation_DataSet\1. Dicom 원본 (Series기준으로, Fold별로)\fold10'
# Slice-wise Label 데이터프레임
path_label_csv = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv'

# Slice 리스트
list_slice_path = glob(path_target + '/*.png')
list_slice_path = sorted(list_slice_path)

# Label DataFrame
df_label = pd.read_csv(path_label_csv)

# 성공/에러 카운트
cnt_suc = 0
cnt_err = 0

# Slice 별로 순회해서 복사
for cur_slice_path in list_slice_path:
    # 현재 Slice 파일명
    cur_slice_fn = os.path.basename(cur_slice_path).replace('.png', '')
    # SeriesID 파싱
    cur_df = df_label[df_label['ID_Slice'] == cur_slice_fn].reset_index(drop=True)
    cur_SeriesID = cur_df['ID_Series'][0]
    # SeriesID에 해당하는 폴더 복사
    if os.path.isdir(os.path.join(path_dest, cur_SeriesID)):
        cnt_err = cnt_err + 1
        print('Error - {} 폴더는 이미 있음'.format(os.path.join(path_dest, cur_SeriesID)))

    else:
        shutil.copytree(os.path.join(path_source, cur_SeriesID), os.path.join(path_dest, cur_SeriesID))
        cnt_suc = cnt_suc + 1
        print('Success - {}'.format(os.path.join(path_dest, cur_SeriesID)))

print('Num Series : {}'.format(cnt_suc + cnt_err))
print('Num Success : {}'.format(cnt_suc))
print('Num Fail : {}'.format(cnt_err))

#
# # Series ID 데이터 뽑아올 root 폴더
# root_data_SeriesID = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID★★★'
#
# # 데이터 복사할 타겟 경로
# # root_for_copy = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage (new)\0. all_series (new)'
# # root_for_copy = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage (new)\0. all_series (non_small)'
# root_for_copy = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_series (new)'
#
#
# # Small Hemorrhage 정보 csv
# # df_merge = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\1_cHS\20220422_cHS Small Hemorrhage 논문용 성능 정리\result_Non_Small_Hemorrhage_merged.csv')
# df_merge = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\Label_Slice_Wise (Small Hemorrhage 2차_검토 후).csv')
#
# # SeriesID List 뽑기
# list_SeriesID = df_merge['ID_Series']
#
# # 리스트 중복제거
# my_set = set(list_SeriesID) #집합set으로 변환
# list_SeriesID = list(my_set) #list로 변환
#
# cnt_suc = 0
# cnt_fail = 0
#
# # SeriesID에 해당하는 폴더 복사
# for cur_SeriesID in list_SeriesID:
#     try:
#         shutil.copytree(os.path.join(root_data_SeriesID, cur_SeriesID), os.path.join(root_for_copy, cur_SeriesID))
#         print('copy success : {}'.format(cur_SeriesID))
#         cnt_suc = cnt_suc + 1
#     except:
#         print('copy fail : {}'.format(cur_SeriesID))
#         cnt_fail = cnt_fail + 1
#
# print('Num Series : {}'.format(cnt_suc + cnt_fail))
# print('Num Success : {}'.format(cnt_suc))
# print('Num Fail : {}'.format(cnt_fail))