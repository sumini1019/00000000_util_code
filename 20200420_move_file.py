# Ischemic 원본 파일들 중, NCCT 폴더 내 파일들만
# 1개 폴더로 옮기고, label csv 만들기 위한 코드


import shutil
import os
import pandas as pd

# 원본 파일 경로
path = 'C:/00000000 Data/Ischemic_Stroke_NCCT/'
path_cur = ''
# 복사할 경로
resultPath = 'C:/00000000 Data/Ischemic_Stroke_NCCT(modified)/'

# 원본 파일 경로 내 폴더 리스트
list = os.listdir(path)

count = 1

df = pd.read_csv('C:/00000000 Data/ncct.csv')

#def getfiles(dirpath):
#    a = [s for s in os.listdir(dirpath)
#         if os.path.isfile(os.path.join(dirpath, s))]
#    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
#    return a

def moveFiles(count):
    # 현재 폴더 경로의, dcm 파일들 리스트
    dcm_list = os.listdir(path_cur)

    for filename in dcm_list:

        # 원본 파일, 복사 파일 경로 설정 후 복사
        fromFilePathName = path_cur + filename
        resultFilePathName = resultPath + filename
        shutil.copy(fromFilePathName, resultFilePathName)

        # label에 대한 data frame 설정 (ischemic 데이터이므로, 모든 label 0으로 설정)

        new_filename = filename.split('.')[0]

        #df.loc[count] = [filename, 0, 0, 0, 0, 0, 0]
        df.loc[count] = [new_filename, 0, 0, 0, 0, 0, 0]

        print(resultFilePathName)

        count += 1      # 행 number

    return count

# 원본 파일 폴더 수만큼 반복
for i in range(1, len(list)+1):
    # 환자 폴더 들어간 후, NCCT 폴더만 접근
    path_cur = path + list[i-1] + "/NCCT/"
    # 파일 복사 및 label 파일 생성 함수 호출
    count = moveFiles(count)

# label dataframe 저장
df.to_csv("C:/00000000 Data/Ischemic_Stroke_NCCT(modified)/label_ischemic_ncct.csv", index=False)