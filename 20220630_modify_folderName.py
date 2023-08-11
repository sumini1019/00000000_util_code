### 목적 ###
# -> 특정 폴더명을 변경

import pandas as pd
import random
import shutil
import os

# 이름 변경할 원본 파일 경로
path_root = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET_Vessel'

# 폴더 리스트
list_dir = os.listdir(path_root)

for cur_dir in list_dir:
    # 변경 전 폴더 경로
    path_target = os.path.join(path_root, cur_dir, 'hemi_original_WB_Vessel_MIP_ALL')
    # 변경 할 폴더명 경로
    path_modify = path_target.replace('hemi_original_WB_Vessel_MIP_ALL', 'hemi_original_WB_Vessel_PNG_MIP_ALL')

    try:
        os.rename(path_target, path_modify)
        print('성공 - {}'.format(path_modify))
    except:
        print('실패 - {}'.format(path_modify))
        continue