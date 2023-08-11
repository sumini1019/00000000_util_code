### 목적 ###
# csv 파일 읽고, 적혀있는 파일 리스트 사용해서,
# 폴더 내 해당 파일들을 삭제

import shutil
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv


# 비교할 2개 csv 로드
df_for_del = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/210518_Annotation변경시_삭제된_파일리스트.csv')

# 파일 검사 후 삭제할 폴더 경로
file_path = "Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Quarter_v2/fold5"

# 폴더 경로의, 서브 폴더들 경로 가지고 오기
file_path_sub = os.listdir(file_path)

cnt_all = 0

# 하위 폴더마다 file 삭제 반복
for sub_path in file_path_sub:
    # path_left = os.path.join(file_path, cur_id.replace('.dcm', '_DMS_LQ.dcm'))
    # path_right = os.path.join(file_path, cur_id.replace('.dcm', '_DMS_RQ.dcm'))
    #
    # if os.path.exists(path_left):
    #     os.remove(path_left)
    # if os.path.exists(path_right):
    #     os.remove(path_right)

    cnt_sub = 0

    # 파일 존재 여부 검사 후 삭제
    for cur_id in df_for_del['file_deleted']:
        path_left = os.path.join(file_path, sub_path, cur_id.replace('.dcm', '_DMS_LQ.dcm'))
        path_right = os.path.join(file_path, sub_path, cur_id.replace('.dcm', '_DMS_RQ.dcm'))

        if os.path.exists(path_left):
            os.remove(path_left)
            cnt_sub = cnt_sub + 1

        if os.path.exists(path_right):
            os.remove(path_right)
            cnt_sub = cnt_sub + 1

    print('{} cnt : {}'.format(sub_path, cnt_sub))

    # 전체 삭제된 개수 누적
    cnt_all = cnt_all + cnt_sub

print('all cnt : {}'.format(cnt_all))