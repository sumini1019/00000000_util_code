import numpy as np
import pandas
import os
import shutil

path_src = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\Prep_DATA'
path_dst = r'C:\Users\user\Downloads\NIA_ASEPCTS 샘플 데이터'


list_src_folder = os.listdir(path_src)

for idx, folder_name in enumerate(list_src_folder[:10]):
    path_src_folder = os.path.join(path_src, folder_name, 'fig_original')
    path_dst_folder = os.path.join(path_dst, folder_name)
    shutil.copytree(path_src_folder, path_dst_folder)

    path_src_file = os.path.join(path_src, folder_name, folder_name + '_ncct.nii.gz')
    path_dst_file = os.path.join(path_dst, folder_name + '_ncct.nii.gz')
    shutil.copy2(path_src_file, path_dst_file)