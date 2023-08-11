### 목적 ###
# 이미지에서, 특정 클래스의 pixel이 있는지 여부를 검사하고,
# pixel 개수와, 이미지 리스트를 뽑기
# 리스트는 csv로 저장

import shutil
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv

# 원본 파일 경로
#path = 'D:/cHS_Test/image'
#path = '//heuronnas/연구개발부/Sumin_Jung/00000000_DATA/1_cHS/20201021_cHS_Annotated/label'
# path = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210202_only_vessel/WithoutS/label'
# path = 'Z:/Sumin_Jung/00000000_DATA/1_cHS/20210218_cHS_Gil_Hemo+Normal(n77)/label_subtype'
path = 'D:/00000000 Code/20210308_EfficientNet_Classification/data/elvo_classify_DMS/elvo/w4080/preprocessed_cropped_padding_margin_15/ground_truth'

# 복사할 경로
# resultPath = '//heuronnas/연구개발부/Sumin_Jung/00000000_DATA/1_cHS/20201021_cHS_Annotated/label_only_hemorrhage'
# resultPath = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210202_only_vessel/WithoutS/label_modified'
resultPath = 'Z:/Sumin_Jung/00000000_DATA/1_cHS/20210218_cHS_Gil_Hemo+Normal(n77)/label_only_hemo'
# 원본 파일 리스트
label_list = os.listdir(path)

# for label_name in label_list:
#         full_filename = os.path.join(path, label_name)
#         ext = os.path.splitext(full_filename)[-1]
#         if ext == '.png':
#             print(full_filename)

# 카운트
count = 0

# 특정 클래스가 존재하는, 파일 리스트 생성
list_class_file_path = []

def threshold_Files(count, id_patient):
    # 원본 파일 아이디
    filename = id_patient

    # 원본, threshold 변환 이미지 경로 설정
    fromFilePathName = path + '/' + filename
    resultFilePathName = resultPath + '/' + filename


    # 원본 이미지 로드 및 np 변환
    src_img = Image.open(fromFilePathName)
    src_img = np.array(src_img)

    # thresholding (원본 이미지에 대해, 0 초과하는 값은, 모두 1로 변환)
    _, threshold_image = cv2.threshold(src_img, 0, 1, cv2.THRESH_BINARY)

    # threshold 이미지 PIL 변환 및 저장
    th = Image.fromarray(threshold_image)
    th.save(resultFilePathName)

    count += 1
    return count


def check_pixel_value(count, id_patient, target_class_pixel_val):
    # 원본 파일 아이디
    filename = id_patient

    # 원본, threshold 변환 이미지 경로 설정
    fromFilePathName = path + '/' + filename
    # resultFilePathName = resultPath + '/' + filename

    # 원본 이미지 로드 및 np 변환
    src_img = Image.open(fromFilePathName)
    src_img = np.array(src_img)

    # 해당 이미지에서, 픽셀값 별 픽셀개수 카운트
    class_num, pixel_counts = np.unique(src_img, return_counts=True)  # , return_index=True)
    # 픽셀값 별 픽셀개수를 dictionary 형태로 변환
    dict_class_pixel_counts = dict(zip(class_num, pixel_counts))
    # 타겟 class의 픽셀 개수 확인
    try:
        num_target_class = dict_class_pixel_counts[target_class_pixel_val]
    except:
        num_target_class = 0

    # 픽셀 개수가 threshold 이상이면, list에 더하기
    if (num_target_class > 0):
        list_class_file_path.append(filename)
        count += 1
    else:
        print('{} dont have class pixel'.format(filename))
    return count

def save_csv_from_list(list_for_save):

    with open('list_csv.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writecolumn(list_for_save)


# dataframe에서 순서대로 파일 이름 파싱
#for i in range(1, len(df)+1):
#    label_name = df.loc[i-1][0]

for label_name in label_list:
    # 현재 파일의 확장자
    ext = os.path.splitext(label_name)[-1]
    if(ext=='.png'):
        # thresholding 함수 호출
        # count = threshold_Files(count, label_name)
        count = check_pixel_value(count, label_name, 1)

print('Num of processed files : ', count)

print(list_class_file_path)

# csv 저장
df_list = pd.DataFrame(list_class_file_path)
df_list.to_csv('list_have_DMS.csv', index=False)