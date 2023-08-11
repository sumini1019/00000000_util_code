# 특정 폴더 내 파일들만 복사


import shutil
import os
import pandas as pd

# 원본 파일 경로
#path = 'C:/00000000 Data/20200327 ICH Data/Top3/input/stage_2_train_images'
path = '//heuronnas/연구개발부/Stroke/DATA/20200727 ICH Train Dataset_n752803/stage_2_train_images'
path_cur = ''
# 복사할 경로
#resultPath = 'C:/00000000 Data/20200327 ICH Data/Top3/input/stage_2_train_images (exclude intact)'
resultPath = '//heuronnas/연구개발부/Stroke/DATA/20200727 ICH Train Dataset_n752803/stage_2_train_images_png'

# 원본 파일 경로 내 파일(or 폴더) 리스트
list = os.listdir(path)

count = 1

# csv 데이터 프레임 저장
df = pd.read_csv('C:/00000000 Data/20200327 ICH Data/Top3/input/rsna_train (exclude intact).csv')

#def getfiles(dirpath):
#    a = [s for s in os.listdir(dirpath)
#         if os.path.isfile(os.path.join(dirpath, s))]
#    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
#    return a

def moveFiles(count, id_patient):
    # 현재 폴더 경로의, dcm 파일들 리스트
    #dcm_list = os.listdir(path_cur)

    #for filename in dcm_list:

    filename = id_patient

    # 원본 파일, 복사 파일 경로 설정 후 복사
    fromFilePathName = path + '/' + filename
    resultFilePathName = resultPath + '/' + filename
    #resultFilePathName = id_patient + '_' + filename
    shutil.copy(fromFilePathName, resultFilePathName)

    print(resultFilePathName)

    count += 1      # 행 number

    return count

# csv 파일 내, 행 개수만큼 반복
for i in range(1, len(df)+1):

    # csv 파일 내, 이미지 이름
    filename_cur = df.loc[i-1].Image
    id_patient = filename_cur + '.dcm'

    # 파일 복사 함수 호출
    count = moveFiles(count, id_patient)

print('Num of copied file : ', count)