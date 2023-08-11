### 목적 ###
# -> image는 있으나, 동일한 json은 없는 파일만 확인 후, 복사

import shutil
import os
import pandas as pd
import math

# 복사할 원본 파일 경로
path_image = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210310_cHS_Annotation_image_and_json\image'
path_json = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210310_cHS_Annotation_image_and_json\json_complete'

path_cur = ''

# 복사할 경로
resultPath = r'H:\20220426_cHS_Re_Annotation'

# 이름만 가지고 올 파일들의 리스트
json_list = os.listdir(path_json)

# Image 중에 annotation 없는 놈은?
image_list = os.listdir(path_image)
no_json_image = []
path_no_json_image = r'H:\20220426_cHS_Re_Annotation\no_json_image'

# json 없는 image 리스트 생성
for cur_image in image_list:
    if not cur_image.replace('.png', '.json') in json_list:
        no_json_image.append(cur_image)

# image 복사
for cur_image in no_json_image:
    shutil.copy(os.path.join(path_image, cur_image), os.path.join(path_no_json_image, cur_image))



# 복사할 원본 파일 경로
path_image = r'E:\20220425_Dataset_Small_Hemorrhage\Small Hemorrhage\5. subdural'
path_exclude = r'E:\20220425_Dataset_Small_Hemorrhage\Small Hemorrhage\제외 case'
path_cur = ''
# 복사할 경로
resultPath = r'E:\20220425_Dataset_Small_Hemorrhage\Small Hemorrhage\5. subdural (new)'
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