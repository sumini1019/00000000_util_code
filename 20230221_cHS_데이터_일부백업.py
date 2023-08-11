# 2023.01.27
# - csv 읽어서, 해당 리스트에 해당하는
# - RSNA Annotation, 탑병원 Annotation 파일 읽기

import pandas as pd
import shutil
import os
from glob import glob
import nrrd

path_src = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_for_Segmentation★★★'
path_dst = r'D:\00000000_Data\TrainSet_for_Segmentation★★★'
import os

# import os
#
# # A 폴더와 B 폴더의 파일 리스트를 각각 저장
# a_files = os.listdir(r'D:\00000000_Data\TrainSet_for_Segmentation★★★\label_binary')
# b_files = os.listdir(r'D:\00000000_Data\TrainSet_for_Segmentation★★★\label_subtype')
#
# # 특정 문자열 제거
# a_files = [s.replace('_label.png', '') for s in a_files]
# b_files = [s.replace('_label.png', '') for s in b_files]
#
#
# # A 폴더와 B 폴더의 파일 리스트를 정렬하여 비교
# a_files_sorted = sorted(a_files)
# b_files_sorted = sorted(b_files)
#
# # 파일 리스트가 동일한지 확인
# if a_files_sorted == b_files_sorted:
#     print('A 폴더와 B 폴더의 파일 리스트가 동일합니다.')
# else:
#     print('A 폴더와 B 폴더의 파일 리스트가 동일하지 않습니다.')

def copy_files(src, dst, count=1000):
    dirs = os.listdir(src)

    for idx, dir in enumerate(dirs):
        if idx <= 2:
            continue

        cur_src_dir = os.path.join(src, dir)
        cur_dst_dir = os.path.join(dst, dir)

        if not os.path.isdir(cur_src_dir):
            continue

        if not os.path.exists(cur_dst_dir):
            os.makedirs(cur_dst_dir)

        files = os.listdir(cur_src_dir)
        files = sorted(files)
        for i, file in enumerate(files):
            cur_path_src = os.path.join(src, dir, file)
            cur_path_dst = os.path.join(dst, dir, file)

            if i >= count:
                break
            shutil.copy2(cur_path_src, cur_path_dst)

for src_dir, dirs, files in os.walk(path_src):
    dst_dir = src_dir.replace(path_src, path_dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    copy_files(src_dir, dst_dir)