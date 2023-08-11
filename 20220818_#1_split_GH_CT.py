### 목적 ###
# - 길병원 CT / MR 데이터를,
# - CTA / CBV / DWI 등으로 데이터 나눠서 삽입 목적


import shutil
import os
import pandas as pd
from glob import glob

# 영상 데이터 타입 리스트
list_type = ['CTA', 'CTP', 'CBV', 'DWI', 'ADC', 'FLAIR']

# 영상 데이터 참조 csv
path_csv = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\GMC_ID_Renewal_List_v220715_Sumin수정중_csv_ver_final.csv'

# 원본 데이터
# path_data = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\길병원 데이터수집 원본'
# path_data = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\길병원 데이터수집 (2차_ID)\길병원 데이터수집 (GMC Reg. ID 기준_ID 중복건_수동으로 하나하나 옮겨야됨;;)'
path_data = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\길병원 데이터수집 (2차_ID)\길병원 데이터수집 (GMC Reg. ID 기준)'

# 옮길 타겟 데이터
# path_target = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\ID_Renewal_Test'
path_target = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\길병원 데이터수집 (2차_ID)\길병원 데이터수집 (ID Renewal 기준)'

# csv dataframe
df = pd.read_csv(path_csv)

# 원본 데이터 폴더 리스트
list_folder = os.listdir(path_data)

# 파일명 error 있는 환자 리스트
list_err_type = []

# DataFrame으로부터, Renew ID 체크
def check_ID(folder_name):
    # 폴더명에서 ID 파싱
    id_old = 'ID_' + folder_name[:8]

    # Dataframe에서, old ID 에 해당하는, new ID 파싱
    list_id_renew = list(df[df['GMC_Reg_ID'] == id_old]['Renewal_ID'])

    return list_id_renew

# 원본 폴더에서, 파일 리스트 내, type명이 잘못된게 있는지 검사
def check_type(list_cur_dcm):
    for cur_file in list_cur_dcm:
        # 파일명에서 type 파싱
        fn_type = os.path.basename(cur_file)[:-8]

        # type이 없을 경우, error 발생
        if not fn_type in list_type:
            print('Type Error - ', cur_file)
            list_err_type.append(cur_file)

for cur_folder in list_folder:
    # 현재 RegID 프린트
    print('Processing - {}'.format(os.path.basename(cur_folder)))

    # 현재 폴더의, ID 확인 (DataFrame에서)
    list_id_renew = check_ID(cur_folder)
    # 강제변경 (ID 중복건들에 대해서.... 수동하기 귀찮아서)          list_id_renew = ['GH1062']
    # 원본 데이터 경로
    path_cur_data = os.path.join(path_data, cur_folder)

    # 원본 데이터 dcm 리스트
    list_cur_dcm = glob(path_cur_data + '/*.dcm')

    # 원본 파일명 중, Type 이름이 잘못된게 있는지 검사
    check_type(list_cur_dcm)

    # ID가 복수개일 경우, 각각 복사
    for id_renew in list_id_renew:
        # 데이터 타입별로 파싱
        for data_type in list_type:
            # 원본 폴더의 모든 dcm 파일 중에서, DataType 에 해당하는 파일만 복사
            for cur_dcm in list_cur_dcm:
                # dcm 파일명
                cur_dcm_fn = os.path.basename(cur_dcm)

                # 복사할 타겟 경로
                path_cur_target = os.path.join(path_target, id_renew, data_type, cur_dcm_fn)

                # 데이터 타입에 해당하는가?
                if data_type in cur_dcm_fn:
                    # 폴더 생성
                    os.makedirs(os.path.dirname(path_cur_target), exist_ok=True)

                    # 해당하면 복사
                    shutil.copyfile(cur_dcm, path_cur_target)

print('*** Type Error 리스트')
print(list_err_type)