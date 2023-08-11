### 목적 ###
# -> 폴더의 파일 이름을 일괄적으로 바꾸는 코드

import shutil
import os
import pandas as pd
import math
from glob import glob

cnt_err = 0

# 타겟 폴더
# path_target = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20220510_VoxelMorph 데이터 기반, DMS-EIC Feature\DMS_WB\WB_DMS_20220511-0914_b4_batch4_lr0.0001_Trans(True)_Freeze0\test\fold0\ep15'
# path_target = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20220510_VoxelMorph 데이터 기반, DMS-EIC Feature\DMS_WB\WB_DMS_20220511-0914_b4_batch4_lr0.0001_Trans(True)_Freeze0\test\fold0\ep25'


### DMS - Hemi - Small (완료)
# path_root = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20220510_VoxelMorph 데이터 기반, DMS-EIC Feature\DMS_Hemi\Hemi_DMS_20220513-1555_Hemi오류_다시학습_b4_batch8_lr0.0001_Trans(True)_Freeze0'

### DMS - Hemi (Train 완료 / Test 완료)
# path_root = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20220510_VoxelMorph 데이터 기반, DMS-EIC Feature\DMS_Hemi\Hemi_DMS_20220513-1555_Hemi오류_다시학습_b4_batch8_lr0.0001_Trans(True)_Freeze0\test\fold0'
# path_root = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20220510_VoxelMorph 데이터 기반, DMS-EIC Feature\DMS_Hemi\Hemi_DMS_20220513-1555_Hemi오류_다시학습_b4_batch8_lr0.0001_Trans(True)_Freeze0\train\fold5'
# list_folder = os.listdir(path_root)

### DMS - WB (Train X / Test 완료)
path_root = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20220510_VoxelMorph 데이터 기반, DMS-EIC Feature\DMS_WB\WB_DMS_20220511-0914_b4_batch4_lr0.0001_Trans(True)_Freeze0\train\fold5'
list_folder = os.listdir(path_root)[3:]






for cur_folder in list_folder:
    # 현재 폴더의 절대 위치
    path_cur_folder = os.path.join(path_root, cur_folder)

    print(path_cur_folder)

    # 현재 위치가, 폴더가 맞는지 확인
    if os.path.isdir(path_cur_folder):
        # 타겟 폴더의 파일 리스트
        list_file = sorted(glob(path_cur_folder +'/*.npz'))

        # 파일 이름 변경
        for cur_file in list_file:
            # 파일의 전체 경로에서, 이름만 파싱
            cur_file_basename_old = os.path.basename(cur_file)

            # 새 이름 설정 (".dcm" 삭제)
            cur_file_basename_new = cur_file_basename_old.replace(".dcm", "")

            try:
                # 이름 변경
                os.rename(cur_file, os.path.join(path_cur_folder, cur_file_basename_new))
                # print('Success - {}'.format(cur_file_basename_new))
            except:
                cnt_err += 1
                print('Error - {}'.format(cur_file_basename_new))


    else:
        continue


    print('Total Error - {}'.format(cnt_err))