### 목적 ###
# -> cHS에서, label 이미지에 대해서, thresholding 하려고
# -> 5개 클래스를, 1개 클래스로 줄이려고
# -> 즉, 픽셀값 0을 제외한 1~255를 모두 1로 바꾸려고

# 폴더의 파일 리스트 뽑고,
# 파일 이름이 같지만, 확장자가 다른 파일들을 모두 복사하는 코드

import shutil
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 원본 파일 경로
#path = 'D:/cHS_Test/image'
#path = '//heuronnas/연구개발부/Sumin_Jung/00000000_DATA/1_cHS/20201021_cHS_Annotated\label'
# path = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210202_only_vessel/WithoutS/label'
path = r'C:\Users\user\Downloads\1. 이어명 인턴 인수인계 자료\1_1. fold10 json result\fold_10_Label_Subtype'
# 복사할 경로
# resultPath = '//heuronnas/연구개발부/Sumin_Jung/00000000_DATA/1_cHS/20201021_cHS_Annotated\label_only_hemorrhage'
# resultPath = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210202_only_vessel/WithoutS/label_modified'
resultPath = r'C:\Users\user\Downloads\1. 이어명 인턴 인수인계 자료\1_1. fold10 json result\fold_10_Label_Binary'
# 원본 파일 리스트
label_list = os.listdir(path)

# for label_name in label_list:
#         full_filename = os.path.join(path, label_name)
#         ext = os.path.splitext(full_filename)[-1]
#         if ext == '.png':
#             print(full_filename)

# 카운트
count = 0
# dataframe 형태 변환
#df = pd.DataFrame(label_list)

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

# dataframe에서 순서대로 파일 이름 파싱
#for i in range(1, len(df)+1):
#    label_name = df.loc[i-1][0]

for label_name in label_list:
    # 현재 파일의 확장자
    ext = os.path.splitext(label_name)[-1]
    if(ext=='.png'):
        # thresholding 함수 호출
        count = threshold_Files(count, label_name)


print('Num of processed files : ', count)