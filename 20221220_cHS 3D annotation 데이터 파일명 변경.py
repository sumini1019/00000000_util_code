### 목적 ###
# -> 3D Annotation 파일명을,
# -> 2D Annotation의 파일 index와 동일하게 변경

import pandas as pd
import random
import shutil
import os

# 원본 파일 경로
path_root_Hemo = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series\Hemo'
path_root_Normal = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series\Normal'

# 복사할 파일 경로
path_root_dest_Hemo = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series(ID index 추가)\Hemo'
path_root_dest_Normal = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series(ID index 추가)\Normal'

os.makedirs(path_root_dest_Hemo, exist_ok=True)
os.makedirs(path_root_dest_Normal, exist_ok=True)

# 참조할 2D Annotation 엑셀 파일
path_excel = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\3. Ground Truth (Index 추가)\cHS_Label_Patient_wise_modified.xlsx'

# 폴더 리스트
list_dir_Hemo = os.listdir(path_root_Hemo)
list_dir_Normal = os.listdir(path_root_Normal)

# 엑셀 파일 로드
df = pd.read_excel(path_excel)


# for idx, cur_dir in enumerate(list_dir_Hemo):
#     # ID 파싱
#     cur_ID = cur_dir.split('.nii')[0]
#
#     # 엑셀에서, ID에 해당하는 index+ID 파싱
#     cur_df = df[df['ID_Series'].str.contains(cur_ID)].reset_index(drop=True)
#     if len(cur_df) == 1:
#         cur_index_ID = cur_df.loc[0, 'ID_Series']
#     else:
#         print('Error - 해당하는 DF가 없음 {} '.format(cur_ID))
#         continue
#
#     # 변경 전 폴더 경로
#     path_target = os.path.join(path_root_Hemo, cur_ID+'.nii.gz')
#     # 변경 할 폴더명 경로
#     path_modify = os.path.join(path_root_dest_Hemo, cur_index_ID+'.nii.gz')
#
#     # 복사
#     try:
#         shutil.copy(path_target, path_modify)
#         # print('성공 - {}'.format(path_modify))
#     except:
#         print('실패 - {}'.format(path_modify))
#         continue


start_idx = 8883
for idx, cur_dir in enumerate(list_dir_Normal):
    # ID 파싱
    cur_ID = cur_dir.split('.nii')[0]

    # # 엑셀에서, ID에 해당하는 index+ID 파싱
    # cur_df = df[df['ID_Series'].cur_folder.contains(cur_ID)].reset_index(drop=True)
    # if len(cur_df) == 1:
    #     cur_index_ID = cur_df.loc[0, 'ID_Series']
    # else:
    #     print('Error - 해당하는 DF가 없음 {} '.format(cur_ID))
    #     continue

    # 8883부터 시작하는 인덱스 생성
    cur_index_ID = "{}_{}".format(start_idx, cur_ID)
    start_idx += 1

    # 변경 전 폴더 경로
    path_target = os.path.join(path_root_Normal, cur_ID+'.nii.gz')
    # 변경 할 폴더명 경로
    path_modify = os.path.join(path_root_dest_Normal, cur_index_ID+'.nii.gz')

    # 복사
    try:
        shutil.copy(path_target, path_modify)
        # print('성공 - {}'.format(path_modify))
    except:
        print('실패 - {}'.format(path_modify))
        continue

# 2023.08.09
# - Normal 파일 리스트 dataframe 생성 및 csv 저장
# 변경된 ID들을 저장할 리스트
# 지정한 경로에서 파일 리스트를 얻기
file_list = os.listdir(path_root_dest_Normal)
# 파일 이름에서 확장자 제거
file_ids = [os.path.splitext(file)[0] for file in file_list]
# 파일 ID들로 데이터프레임 생성
df_ids = pd.DataFrame(file_ids, columns=['ID_Series'])
# csv 파일로 저장
df_ids.to_csv('changed_ids.csv', index=False)