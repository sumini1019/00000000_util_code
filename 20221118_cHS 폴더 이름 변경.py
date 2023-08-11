### 목적 ###
# -> 특정 폴더명을 변경

import pandas as pd
import random
import shutil
import os

# 이름 변경할 원본 파일 경로
path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\1. 2D Annotation (일부)\1. Data_Series\Hemo'

# 폴더 리스트
list_dir = os.listdir(path_root)

idx = 0
str_idx = ''
for cur_dir in list_dir:
    # 현재 index
    idx = idx + 1
    # index 문자 변환 (4자리수)
    if idx < 10:
        str_idx = '000' + str(idx)
    elif idx < 100:
        str_idx = '00' + str(idx)
    elif idx < 1000:
        str_idx = '0' + str(idx)
    else:
        str_idx = str(idx)

    # 변경 전 폴더 경로
    path_target = os.path.join(path_root, cur_dir)
    # 변경 할 폴더명 경로
    path_modify = path_target.replace(cur_dir, str_idx+'_'+cur_dir)

    try:
        os.rename(path_target, path_modify)
        print('성공 - {}'.format(path_modify))
    except:
        print('실패 - {}'.format(path_modify))
        continue