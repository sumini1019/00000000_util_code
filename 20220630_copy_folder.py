### 목적 ###
# -> 특정 폴더만 복사

import pandas as pd
import random
import shutil
import os

# 복사할 경로
path_dest = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET_Vessel'

# 복사할 원본 파일 경로
path_root = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET_Vessel_Old'

# 폴더 리스트
list_dir = os.listdir(path_root)

for cur_dir in list_dir:
    path_target = os.path.join(path_root, cur_dir, 'hemi_original_WB_Vessel')
    path_dest_copy = os.path.join(path_dest, cur_dir, 'hemi_original_WB_Vessel')

    try:
        shutil.copytree(path_target, path_dest_copy)
        print('성공 - {}'.format(path_dest_copy))
    except:
        print('실패 - {}'.format(path_dest_copy))
        continue