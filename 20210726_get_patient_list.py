
### 목적 ###
# -> 폴더 내 파일 리스트에서, 환자 번호만 뽑아내는 코드

import pandas as pd
import random
import shutil
import os

def get_patient_list(path):
    list_file = os.listdir(path)
    list_file = sorted(list_file)

    for i in range(0, len(list_file)):
        if i < 452:
            list_file[i] = 'AJ0' + list_file[i][:3]
        else:
            list_file[i] = list_file[i][:6]

    # 중복 제거
    list_file_set = set(list_file)
    list_file = sorted(list(list_file_set))

    # # csv 변환 및 저장
    # df_list_file = pd.DataFrame(list_file)  # , columns='patient_id')
    # df_list_file.to_csv('fold0_list_patient.csv', index=False)

    #

    return list_file

# 이미지 DCM 폴더
path_dcm = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\dcm'
# 이미지 DCM 아이디별로 나눌 폴더
path_dcm_split = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\dcm_by_id'

# 환자 번호 리스트
patient_list = get_patient_list(path_dcm)

# 원본 파일 리스트
list_file = os.listdir(path_dcm)
list_file = sorted(list_file)

# ID 별 스플릿
for cur_patient in patient_list[275:]:
    # 아주대 / 길병원 Patient 별, 파일의 index 파싱
    if 'AJ' in cur_patient:
        old_id = cur_patient[3:]
        # 원본 중, old_id 가 속한 파일의 인덱스 파싱
        list_idx_file = [i for i in range(len(list_file)) if old_id in list_file[i][:3]]
    else:
        old_id = cur_patient
        # 원본 중, old_id 가 속한 파일의 인덱스 파싱
        list_idx_file = [i for i in range(len(list_file)) if old_id in list_file[i]]

    # 해당 인덱스 파일들을 카피
    for cur_idx in list_idx_file:
        # 카피할 파일의 경로
        path_copy_file = os.path.join(path_dcm, list_file[cur_idx])

        # 카피 될 경로의 폴더 생성
        if not os.path.isdir(os.path.join(path_dcm_split, cur_patient)):
            os.mkdir(os.path.join(path_dcm_split, cur_patient))

        # 카피 파일의 목적지 경로
        path_copy_file_dest = os.path.join(path_dcm_split, cur_patient, list_file[cur_idx])

        # 카피
        shutil.copyfile(path_copy_file, path_copy_file_dest)


