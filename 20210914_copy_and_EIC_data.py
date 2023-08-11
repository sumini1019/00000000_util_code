
### 목적 ###
# -> 특정 폴더 내에, 환자 별로 데이터가 나뉘어져 있음
# -> 해당 데이터를, csv 기반으로, DMS / Normal 로 나눠서 저장

import pandas as pd
import random
import shutil
import os

################### DMS 파일들만, fold별 복사

# 복사할 경로
resultPath_eic = r'Z:\Stroke\SharingFolder\20210915_cELVO_길병원_학습용데이터\used_for_dms_0'
# resultPath_dms = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/dms(for_test)'
# resultPath_normal = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210720_LDCT_변환데이터/fold5/normal(for_test)'

# 복사할 원본 파일 경로
# path = 'Z:/Stroke/cASPECTS/LearningSET_DECTv2/Hemispheric_Comparison/LH'
path_left = 'Z:/Stroke/TestData/TrainSET_cELVO_v210913'    # + '\{환자아이디}\ELVO\LH
path_right = 'Z:/Stroke/TestData/TrainSET_cELVO_v210913'   # + '\{환자아이디}\ELVO\RH

# 환자 리스트
# df_patient_list = pd.read_csv("Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210726_LDCT변환+Normalization/fold_split/fold5_list_patient.csv")

# csv 데이터 프레임 로드
df = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20210915_LDCT_Norm_아주대_길병원_추가\df_train_all_아주대_길병원.csv')
df = df[11122:]
df = df[df['USED for DMS'] == 0]
# 인덱스 초기화
df.reset_index(drop=True, inplace=True)

# csv 파일 내, 행 개수만큼 반복
for i in range(0, len(df)):

    # 방향 상관없는 파일 이름 파싱
    name_file = df.loc[i].ID_Filename

    # Left / Right 파일 네이밍
    name_file_left = name_file.replace('HR-', 'HR_') + '-LH.dcm'
    name_file_right = name_file + '-RH.dcm'

    # path_file_left = os.path.join(path_left, df.loc[i].ID) + '/ELVO/LH/{}'.format(name_file_left)
    path_file_left = os.path.join(path_left, df.loc[i].ID, 'ELVO', 'LH', name_file_left)
    path_file_right = os.path.join(path_right, df.loc[i].ID, 'ELVO', 'RH', name_file_right)

    # Left
    try:
        shutil.copy(path_file_left,
                    os.path.join(resultPath_eic, name_file_left.replace('HR_', 'HR-NCCT_{}_'.format(df.loc[i].ID))))
    except:
        print("False Copy : {}".format(name_file_left))

    # Right도 똑같이 적용
    try:
        shutil.copy(path_file_right, os.path.join(resultPath_eic, name_file_right.replace('HR-', 'HR-NCCT_{}_'.format(df.loc[i].ID))))
    except:
        print("False Copy : {}".format(name_file_right))