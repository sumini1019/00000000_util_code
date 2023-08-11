
### 목적 ###
# -> 폴더 내 파일 리스트에서, 환자 번호만 뽑아내는 코드

import pandas as pd
import random
import shutil
import os

def cnt_file_num(path):
    list_file = os.listdir(path)

    for i in range(0, len(list_file)):
        list_file[i] = list_file[i][:3]

    # 중복 제거
    list_file_set = set(list_file)
    list_file = list(list_file_set)

    # 정렬
    list_file.sort()

    # csv 변환 및 저장
    df_list_file = pd.DataFrame(list_file)  # , columns='patient_id')
    df_list_file.to_csv('fold0_list_patient.csv', index=False)

# root 경로
path_root = 'Z:/Sumin_Jung/00000000_DATA/4_cELVO(Feautre_embedding)/20210825_테스트용_길병원_LVO환자데이터,길병원Normal_LDCT_Normalization변환/cELVO_GIL_TestSET_v210824'

# sub 폴더 불러들이기
list_folder = os.listdir(path_root)

# 최대 개수
max_folder_num = 0

for cur_folder in list_folder:
    cur_folder_num_file = len(os.listdir(os.path.join(path_root, cur_folder, 'elvo/lh')))
    print(cur_folder, ': ', cur_folder_num_file)
    # 최대 개수 업데이트
    if cur_folder_num_file > max_folder_num:
        max_folder_num = cur_folder_num_file


print('max num: ', max_folder_num)
