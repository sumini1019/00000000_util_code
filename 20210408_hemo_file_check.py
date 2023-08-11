### 목적 ###
# 전체 슬라이스 리스트에서,
# Hemorrhage 슬라이스 (json이 있는 파일) 만 체크해서 csv 만들어주기

import shutil
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv

# 전체 슬라이스 경로
all_slice_path = 'Z:/Sumin_Jung/00000000_DATA/1_cHS/20210119_cHS_Gil_Data/PreProcessed/image_png (labeling 목적의, SSCT png 변환 이미지)'

# Hemo 슬라이스 경로
hemo_slice_path = 'Z:/Sumin_Jung/00000000_DATA/1_cHS/20210119_cHS_Gil_Data/PreProcessed/label_subtype'


# 전체 슬라이스 리스트
list_all_slice = os.listdir(all_slice_path)
# 전체 슬라이스 리스트 dataframe 변환
df_all_slice = pd.DataFrame(list_all_slice, columns=['image'])
# hemo 여부 열 추가
df_all_slice.insert(1, 'hemo', 0)

# Hemo 슬라이스 리스트
list_hemo_slice = os.listdir(hemo_slice_path)
# Hemo 슬라이스 리스트 dataframe 변환
df_hemo_slice = pd.DataFrame(list_hemo_slice, columns=['hemo_slice'])

# 전체 슬라이스의 image 이름 중, hemo 슬라이스도 존재하면, Hemo 여부에 1로 저장
for cur_id in df_all_slice['image']:
    # hemo slice가 있다면,
    if (df_hemo_slice['hemo_slice']==cur_id.replace('.png', '_label.png')).any():
        # Hemo Slice가 있는 index 위치를 반환
        idx_hemo = df_all_slice.index[df_all_slice['image'] == cur_id].tolist()[0]
        # 'Hemo' 열에 대한 값을 1로 변경
        df_all_slice.loc[idx_hemo, 'hemo'] = 1

        print('exist : {}'.format(cur_id))
    else:
        print('not exist : {}'.format(cur_id))

# 저장
df_all_slice.to_csv('List_Slice_isHemo.csv', index=False)