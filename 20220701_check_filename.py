### 목적 ###
# -> 두 폴더 간 상이한 파일명이 있는지 확인

import pandas as pd
import random
import shutil
import os

# 이름 변경할 원본 파일 경로
path_image = r'H:\20220614_이어명 인턴 전달\3. cHS_Segmentation_Dataset\image_png'
path_label = r'H:\20220614_이어명 인턴 전달\3. cHS_Segmentation_Dataset\label_binary'

# 폴더 리스트
list_image = sorted(os.listdir(path_image))
list_label = sorted(os.listdir(path_label))

print('Image Num :', len(list_image))
print('Label Num :', len(list_label))

for root, dirs, files in os.walk(path_image):
    # print(files)

    for cur_file in files:
        fn_label = cur_file.replace('_img.png', '_label.png')

        # label 파일 있나?
        if fn_label in list_label:
            continue
        else:
            print('Label 없음 - {}'.format(fn_label))
