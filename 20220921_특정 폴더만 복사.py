### 목적 ###
# -> 특정 폴더만 복사

import pandas as pd
import random
import shutil
import os

# 복사할 경로
path_root_dst = r'D:\DATA_cELVO_cASPECTS'
# 복사할 원본 파일 경로
path_root_src = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS'

# 기관 리스트
list_hospital = ['1_AJUMC', '2_GMC', '3_EUMC', '4_CSUH', '5_CNUSH', '6_SCHMC', '7_etc']
# 복사할 폴더 리스트
list_folder_to_copy = ['LVO_UL_affined', 'LVO_UL_affined_PADENET']


os.makedirs(path_root_dst, exist_ok=True)


for cur_hospital in list_hospital:
    cur_path_hospital_src = os.path.join(path_root_src, cur_hospital, 'Prep_DATA')
    cur_path_hospital_dst = os.path.join(path_root_dst, cur_hospital, 'Prep_DATA')

    # 병원의 환자 폴더 순회
    list_patient = os.listdir(cur_path_hospital_src)

    for cur_patient in list_patient:
        # 두개 폴더 복사
        for target_folder in list_folder_to_copy:
            try:
                path_src = os.path.join(cur_path_hospital_src, cur_patient, target_folder)
                path_dst = os.path.join(cur_path_hospital_dst, cur_patient, target_folder)

                os.makedirs(os.path.dirname(path_dst), exist_ok=True)
                shutil.copytree(path_src, path_dst)
            except:
                print('복사 실패 : {}'.format(path_src))
                continue


