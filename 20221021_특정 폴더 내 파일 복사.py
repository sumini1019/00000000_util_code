### 목적 ###
# -> 특정 폴더만 복사

import pandas as pd
import random
import shutil
import os

# 복사할 폴더명
list_folder = ['AJ0106', 'AJ0191','AJ0228','AJ0230','AJ0244','AJ0253','AJ0274','AJ0279','AJ0282','AJ0285','AJ0291','AJ0306','AJ0312','AJ0315','AJ0324','AJ0325','AJ0333','AJ0353','AJ0369','AJ0374','AJ0381','AJ0383','AJ0395','AJ0406','AJ0409','AJ0412','AJ0425','AJ0439','AJ0441','AJ0443','AJ0462','AJ0464','AJ0467','AJ0472','AJ0477','AJ0494','AJ0496','AJ0535','AJ0539','AJ0543','AJ0548','AJ0553']
# 복사할 root 경로
path_root_src = r'Z:\Stroke\cELVO\ELVO_Eye_Lens_Dataset_Modify'
# 복사할 경로
path_root_dst = r'Z:\Stroke\cELVO\GT_GUI_2022\Eye_Dev\AffineSET'

os.makedirs(path_root_dst, exist_ok=True)


for cur_folder in list_folder:
    cur_path_src = os.path.join(path_root_src, cur_folder, 'affined_ncct.nii.gz')
    # cur_path_dst = os.path.join(path_root_dst, cur_folder, '_affined_ncct.nii.gz'.format(cur_folder))
    cur_path_dst = os.path.join(path_root_dst, '{}_affined_ncct.nii.gz'.format(cur_folder[3:]))

    try:
        shutil.copyfile(cur_path_src, cur_path_dst)
    except:
        print('복사 실패 : {}'.format(cur_path_dst))
        continue


