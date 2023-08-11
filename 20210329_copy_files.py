### 목적 ###
# -> csv를 통해, 환자 ID를 가져오고,
# -> 해당 환자 ID 폴더만 복사하는 코드



import shutil
import os
import pandas as pd

# 복사할 원본 파일 경로
path = 'z:/Sumin_Jung/00000000_DATA/3_cELVO/20210503_DMS_반구Crop/0. Original Data(Left,Right 폴더분류)/CropRange/RightH'
path_cur = ''
# 복사할 경로
resultPath = 'z:/Sumin_Jung/00000000_DATA/3_cELVO/20210503_DMS_반구Crop/dms'

# csv 데이터 프레임 로드
df = pd.read_csv('z:/Sumin_Jung/00000000_DATA/3_cELVO/20210503_DMS_반구Crop/Annotation_list_FH.csv')

# csv 파일 내, 행 개수만큼 반복
for i in range(1, len(df)+1):
    # csv 파일 내, 이미지 이름
    # name_folder = df.loc[i-1][0]



    try:
        if df.loc[i - 1].Patient_No < 10:
            name_file = '00' + str(df.loc[i - 1].Patient_No) + '_' + df.loc[i - 1].Image + '_' + df.loc[
                i - 1].Side + '.dcm'
        elif df.loc[i - 1].Patient_No < 100:
            name_file = '0' + str(df.loc[i - 1].Patient_No) + '_' + df.loc[i - 1].Image + '_' + df.loc[
                i - 1].Side + '.dcm'
        elif df.loc[i - 1].Patient_No < 1000:
            name_file = str(df.loc[i - 1].Patient_No) + '_' + df.loc[i - 1].Image + '_' + df.loc[i - 1].Side + '.dcm'
        else:
            print('Patient No Error')

        # 파일 복사
        # shutil.copyfile(os.path.join(path, name_file), os.path.join(resultPath, name_file))
        # 파일 이동
        shutil.move(os.path.join(path, name_file), os.path.join(resultPath, name_file))
        # 폴더 복사
        # shutil.copytree(os.path.join(path, name_folder), os.path.join(resultPath, name_folder))
        print("Succesee Copy : {}".format(name_file))
    except:
        print("False Copy : {}".format(name_file))