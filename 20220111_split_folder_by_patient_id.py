### 목적 ###
# -> 파일 이름 기반으로, ID별 폴더 나누기

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


# 환자 ID 리스트
patient_list = get_patient_list(r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\dcm')

# 환자 ID별 폴더 split
