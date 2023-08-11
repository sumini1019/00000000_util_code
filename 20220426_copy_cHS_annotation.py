### 목적 ###
# -> cHS의, Image / Json 파일을 Fold 나눠서 복사하는 코드

import shutil
import os
import pandas as pd
import math

# 복사할 원본 파일 경로
path_image = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210310_cHS_Annotation_image_and_json\image'
path_json = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210310_cHS_Annotation_image_and_json\json_complete'

path_cur = ''

# 복사할 경로
resultPath = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210310_cHS_Annotation_image_and_json\20220426_cHS_Annotation_Fold별 재구성 및 재평가'

# 한 Fold당 몇개 데이터 들어갈지?
num_fold_image = 10000

# 이름만 가지고 올 파일들의 리스트
json_list = os.listdir(path_json)

# Fold 개수 (올림 연산)
num_fold = math.ceil(len(json_list) / num_fold_image)

# Fold별 json 리스트 생성
fold_json = []
for cur_fold in range(0, num_fold):
    # Fold Image 개수만큼 슬라이싱해서 저장
    fold_json.append(json_list[(cur_fold*num_fold_image):(cur_fold+1)*num_fold_image])

# Fold 개수만큼 반복
for idx_fold in range(4, len(fold_json)):
    # 현재 Fold의 저장 폴더 경로
    resultPath_fold = os.path.join(resultPath, 'fold_{}'.format(idx_fold))

    # 폴더 생성
    os.makedirs(resultPath_fold, exist_ok=True)

    # 현재 Fold의, Image / json 복사
    for cur_json in fold_json[idx_fold]:
        # json 복사
        shutil.copy(os.path.join(path_json, cur_json), os.path.join(resultPath_fold, cur_json))

        # image 복사
        filename_image = cur_json.replace('.json', '.png')
        shutil.copy(os.path.join(path_image, filename_image), os.path.join(resultPath_fold, filename_image))