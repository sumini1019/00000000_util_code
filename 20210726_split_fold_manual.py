
### 목적 ###
# -> 환자의 슬라이스 별 probability 값을 사용하여,
# -> 환자 별 dms 여부 결정

import pandas as pd
import random
import shutil
import os

################### DMS 파일들만, fold별 복사

# 복사할 경로
resultPath_dms = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210726_LDCT변환+Normalization/fold5/dms'
resultPath_normal = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210726_LDCT변환+Normalization/fold5/normal'
# resultPath_dms = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/dms(for_test)'
# resultPath_normal = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/normal(for_test)'

# 복사할 원본 파일 경로
# path = 'Z:/Stroke/cASPECTS/LearningSET_DECTv2/Hemispheric_Comparison/LH'
path_left = 'Z:/Stroke/cASPECTS/LearningSET_DECTv2/Hemispheric_Comparison/LH'
path_right = 'Z:/Stroke/cASPECTS/LearningSET_DECTv2/Hemispheric_Comparison/RH'

# 환자 리스트
df_patient_list = pd.read_csv("Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210726_LDCT변환+Normalization/fold_split/fold5_list_patient.csv")

# csv 데이터 프레임 로드
df = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210726_LDCT변환+Normalization/DATA_Annotation_list.csv')
df = df[df['USED for DMS'] == 1]

# 인덱스 초기화
df.reset_index(drop=True, inplace=True)

# 전체 환자 Annotation 리스트에서, 특정 Fold의 환자만 파싱
# 특정 fold의 데이터만 저장할 df 생성
df_fold = df[:0].copy()

list_patient = list(df_patient_list['ID'])

for cur_patient_id in list_patient:
    df_temp = df[df['ID'] == cur_patient_id]

    #df_fold.append(df_temp)
    #pd.merge(df_fold, df_temp, how='outer',on='ID')
    df_fold = df_fold.append(df_temp)

# 기존 DF 대체
df = df_fold.copy()
# 인덱스 초기화
df.reset_index(drop=True, inplace=True)

# csv 파일 내, 행 개수만큼 반복
for i in range(0, len(df)):

    # 방향 상관없는 파일 이름 파싱
    name_file = df.loc[i].ID_Filename

    # Left / Right 파일 네이밍
    name_file_left = name_file.replace('.dcm', '_LH.dcm')
    name_file_right = name_file.replace('.dcm', '_RH.dcm')

    # Left가 DMS 라면, 파일 이동
    if df.loc[i]["DMS - L"] == 1:
        try:
            shutil.copy(os.path.join(path_left, name_file_left), os.path.join(resultPath_dms, name_file_left))
            # print("Succesee Copy : {}".format(name_file_left))
        except:
            print("False Copy : {}".format(name_file_left))
    # DMS가 아니라면, normal에 copy
    else:
        try:
            shutil.copy(os.path.join(path_left, name_file_left), os.path.join(resultPath_normal, name_file_left))
            # print("Succesee Copy : {}".format(name_file_left))
        except:
            print("False Copy : {}".format(name_file_left))

    # Right도 똑같이 적용
    if df.loc[i]["DMS - R"] == 1:
        try:
            shutil.copy(os.path.join(path_right, name_file_right), os.path.join(resultPath_dms, name_file_right))
            # print("Succesee Copy : {}".format(name_file_right))
        except:
            print("False Copy : {}".format(name_file_right))
    # DMS가 아니라면, normal에 copy
    else:
        try:
            shutil.copy(os.path.join(path_right, name_file_right), os.path.join(resultPath_normal, name_file_right))
            # print("Succesee Copy : {}".format(name_file_right))
        except:
            print("False Copy : {}".format(name_file_right))