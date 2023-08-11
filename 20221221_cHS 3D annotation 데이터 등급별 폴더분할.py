### 목적 ###
# -> excel 파일 참조해서,
# -> 아형 / 등급 기준으로 폴더 만들어서 넣기

import pandas as pd
import random
import shutil
import os

# 원본 파일 경로
path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series(ID index 추가)\Hemo'
# 결과 저장 위치
path_save_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series(Subtype, 등급별 분류)'
# 분류 참조 excel 저장 위치
path_excel_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20221221_탑병원 RSNA 데이터 등급분류 결과'

# Subtype 유형
list_subtype = ['EDH', 'ICH', 'IVH', 'SAH', 'SDH']
# 등급 유형
list_grade = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Subtype + 등급에 해당하는 폴더 생성 (35개)
for idx_subtype, cur_subtype in enumerate(list_subtype):
    for idx_grade, cur_grade in enumerate(list_grade):
        path_for_make = os.path.join(path_save_root, cur_subtype + '_' + cur_grade)
        os.makedirs(path_for_make, exist_ok=True)

# 폴더 리스트
list_dir = os.listdir(path_root)

# Subtype 별로 순차적으로 Grade 분류 진행
for idx_subtype, cur_subtype in enumerate(list_subtype):
    # 참조 excel
    df = pd.read_excel(os.path.join(path_excel_root, 'heuron-ICH-{}.xlsx'.format(cur_subtype)))

    # 행 별로 진행
    cnt = 0
    lst_error = []
    for idx, row in df.iterrows():
        try:
            # 1. 최종등급 확인
            cur_grade = row['최종등급']
            # 2. ID 확인
            cur_ID = row['ID_Series'][:-3]
            # 3. 원본을 해당하는 폴더로 복사
            path_ori = os.path.join(path_root, cur_ID + '.nii.gz')
            path_dest = os.path.join(path_save_root, cur_subtype + '_' + cur_grade, cur_ID + '.nii.gz')
            shutil.copy(path_ori, path_dest)
            cnt = cnt+1
        except:
            print('Error - {} / {}'.format(idx, row['ID_Series'][:-3]))
            lst_error.append(row['ID_Series'][:-3])

    # 성공 개수, 에러리스트 확인
    print('*** Subtype : {}'.format(cur_subtype))
    print('* 성공 - {}/{}'.format(cnt, len(df)))
    print('* 에러리스트\n{}'.format(lst_error))
