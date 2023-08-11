### 목적 ###
# - 두 폴더를 merge 하는 작업
# - 1개 폴더를, 다른 폴더에 copy 하여 merge
# - 각 patient 폴더에서, 중복되는 폴더가 있을 수 있음
# -    -> 이 경우, 폴더명에 '_추가수집_20220810' 붙여서 폴더 생성 및 복사






import shutil
import os
import pandas as pd

# 원본 폴더 경로
path_ori = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\길병원 데이터수집 (2차_ID473~)\길병원 데이터수집 (ID Renewal 기준)'

# 복사할 폴더
path_dest = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\ID_Renewal'

# 2022.08.10 기준으로, GH0472 까지만 복사
list_dir_ori = sorted(os.listdir(path_ori))[91:]      # 008 부터 테스트

# 원본 폴더 순으로 복사
for cur_dir in list_dir_ori:

    # 해당 Patient의 폴더 리스트 (영상 Type별)
    list_dir_type = os.listdir(os.path.join(path_ori, cur_dir))

    # 영상 Type별로 반복
    for cur_type in list_dir_type:

        # 원본 경로 / 복사 목적지 경로
        path_cur_ori = os.path.join(path_ori, cur_dir, cur_type)
        path_cur_dest = os.path.join(path_dest, cur_dir, cur_type)

        # Type이 CBV거나, CTP인 경우, 복사 목적지의 폴더명 다르게 변경 (김도현 박사님 요청)
        if cur_type == 'CBV':
            # 1. CBV :   ../CTP/CBV  에 복사
            path_cur_dest = os.path.join(path_dest, cur_dir, 'CTP', cur_type)
            # 2. CTP :   ../CTP/PENUMBRA  에 복사
        elif cur_type == 'CTP':
            path_cur_dest = os.path.join(path_dest, cur_dir, 'CTP', 'PENUMBRA')



        # # 복사할 목적지의 폴더가 이미 존재할 때, 목적지 폴더명에 접미어 붙여서 변경하기 (폐기)
        # if os.path.isdir(path_cur_dest):
        #     path_cur_dest = os.path.join(path_dest, cur_dir, (cur_type + "_추가수집_20220810"))

        # 복사할 목적지의 폴더가 이미 존재하고, 안에 파일이 있다면, 복사하지 않기
        if os.path.isdir(path_cur_dest) and (len(os.listdir(path_cur_dest)) > 0):
            continue
        # 안에 파일이 없다면, 해당 폴더에 복사
        else:
            try:
                shutil.copytree(path_cur_ori, path_cur_dest)
                print('성공 - {}'.format(path_cur_dest))
            except:
                print('실패 - {}'.format(path_cur_dest))
                continue