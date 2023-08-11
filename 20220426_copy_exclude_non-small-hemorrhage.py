### 목적 ###
# -> cHS Small Hemorrhage 데이터셋에서, 제외된 케이스 없애고, 새로운 데이터셋 생성

import shutil
import os
import pandas as pd
import math

# 복사할 원본 파일 경로
path_image = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_slice_dcm'
path_exclude = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\제외 case'
path_cur = ''
# 복사할 경로
resultPath = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_slice_dcm (new)'
os.makedirs(resultPath, exist_ok=True)

# 이름만 가지고 올 파일들의 리스트
exclude_list = os.listdir(path_exclude)
# Image 중에 annotation 없는 놈은?
image_list = os.listdir(path_image)

no_json_image = []
# json 없는 image 리스트 생성
for cur_image in image_list:
    if not cur_image in exclude_list:
        no_json_image.append(cur_image)
# image 복사
for cur_image in no_json_image:
    shutil.copy(os.path.join(path_image, cur_image), os.path.join(resultPath, cur_image))