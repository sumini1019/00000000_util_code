### 목적 ###
# -> root의 sub 폴더들 검사해서,
# -> 파일명이 동일한, 3개 파일이 모두 있는지 검사

import pandas as pd
import random
import shutil
import os

# 이름 변경할 원본 파일 경로
path_root = r'C:\Users\user\Desktop\AJ_Gil_raw_mask'

# 확인하고자 하는 파일명
fn_to_check = ['mask.nii', 'ncct.nii', 'mask_eye.nrrd']

# 폴더 리스트
list_folder = sorted(os.listdir(path_root))

print('Folder Num : {}\n'.format(len(list_folder)))

for root, dirs, files in os.walk(path_root):


    for cur_dir in dirs:
        # 현재 폴더의 파일 리스트
        list_cur_file = os.listdir(os.path.join(root, cur_dir))

        # 파일 3개인지?
        if len(list_cur_file) != 3:
            print('파일 3개 아님 - {}'.format(cur_dir))

        # 파일명 3개 존재하는지 확인
        for check_fn in fn_to_check:
            if not check_fn in list_cur_file:
                print('파일 안 맞음 - {}'.format(cur_dir))
                print('없는 파일 - {}'.format(check_fn))
                print('File list - {}'.format(list_cur_file))
                print('\n')