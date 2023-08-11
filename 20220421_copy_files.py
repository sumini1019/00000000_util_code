### 목적 ###
# -> cHS에서, 특정 아이디를 가진 DCM 파일만 복사하려고 만든 코드

# 폴더의 파일 리스트 뽑고,
# 파일 이름이 같지만, 확장자가 다른 파일들을 모두 복사하는 코드



import shutil
import os
import pandas as pd

# 복사할 원본 파일 경로
path = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_only_hemorrhage\image_dcm'

path_cur = ''

# 복사할 경로
resultPath = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_slice_dcm'

# 이름만 가지고올 파일 경로
jsonPath = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_slice_viz'

# 이름만 가지고 올 파일들의 리스트
img_list = os.listdir(jsonPath)

# 복사할 파일 리스트에서, 확장자 및 이름 변경
for i in range(len(img_list)):
    # img_list[i] = os.path.splitext(img_list[i])[0].replace('_img', '') + '.png'
    # img_list[i] = os.path.splitext(img_list[i])[0].replace('._img', '') + '.dcm'
    # img_list[i] = img_list[i].replace('.png', '_crop_vessel.png')
    img_list[i] = img_list[i].replace('_label_viz.png', '.dcm')

count = 0

# csv 데이터 프레임 저장
#df = pd.read_csv('C:/00000000 Data/20200327 ICH Data/Top3/input/rsna_train (exclude intact).csv')
df = pd.DataFrame(img_list)

def copyFiles(count, id_patient):

    filename = id_patient

    # 원본 파일, 복사 파일 경로 설정 후 복사
    fromFilePathName = path + '/' + filename
    resultFilePathName = resultPath + '/' + filename
    if os.path.isfile(fromFilePathName):
        shutil.copy(fromFilePathName, resultFilePathName)
        # shutil.move(fromFilePathName, path + '/complete/' + filename)
        print("Copy : {}".format(resultFilePathName))
    else:
        print("No Exist : {}".format(resultFilePathName))



    count += 1      # 행 number

    return count

def moveFiles(count, id_patient):
    # 현재 폴더 경로의, dcm 파일들 리스트
    filename = id_patient

    # 원본 파일, 복사 파일 경로 설정 후 복사
    fromFilePathName = path + '/' + filename
    resultFilePathName = resultPath + '/' + filename
    #resultFilePathName = id_patient + '_' + filename
    if(os.path.isfile(fromFilePathName)):
        shutil.move(fromFilePathName, resultFilePathName)
        print('Move to : ', resultFilePathName)
    else:
        print('Cant Found File : ', resultFilePathName)

    count += 1      # 행 number

    return count

# csv 파일 내, 행 개수만큼 반복
for i in range(1, len(df)+1):

    # csv 파일 내, 이미지 이름
    error_img_name = df.loc[i-1][0]
    # id_patient = filename_cur + '.dcm'

    # 파일 복사 함수 호출
    count = copyFiles(count, error_img_name)

print('Num of copied file : ', count)