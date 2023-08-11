### 목적 ###
# -> 참조 폴더에서, 파일 이름 리스트 뽑고,
# -> 원본 폴더에서, 리스트에 해당하지 않는 파일은 삭제하는 코드



import shutil
import os
import pandas as pd

# 원본 파일 경로 (해당 경로에서, 파일 삭제할 것)
path = 'Z:/Sumin_Jung/00000000_DATA/1_cHS/20210120_cHS_RSNA_only_hemorrhage(n107934)_dcm/image_dcm(old)'
path_cur = ''
# 복사할 경로
resultPath = 'Z:/Sumin_Jung/00000000_DATA/1_cHS/20210120_cHS_RSNA_only_hemorrhage(n107934)_dcm/image_dcm'
# 이름만 가지고올 파일 경로
ref_Path = 'D:/00000000_Data/hemorrhage_binary/image'

# 원본 파일 리스트
img_list = os.listdir(ref_Path)

# ref_Path의 파일 이름과 동일한 image 리스트 생성
# 파일 리스트에서, 특정 문자열 제거하여
for i in range(len(img_list)):
    img_list[i] = os.path.splitext(img_list[i])[0].replace('_img', '') + '.dcm'

count = 0

# csv 데이터 프레임 저장
#df = pd.read_csv('C:/00000000 Data/20200327 ICH Data/Top3/input/rsna_train (exclude intact).csv')
df = pd.DataFrame(img_list)

#def getfiles(dirpath):
#    a = [s for s in os.listdir(dirpath)
#         if os.path.isfile(os.path.join(dirpath, s))]
#    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
#    return a

def copyFiles(count, id_patient):
    # 현재 폴더 경로의, dcm 파일들 리스트
    #dcm_list = os.listdir(path_cur)

    #for filename in dcm_list:

    filename = id_patient

    # 원본 파일, 복사 파일 경로 설정 후 복사
    fromFilePathName = path + '/' + filename
    resultFilePathName = resultPath + '/' + filename
    #resultFilePathName = id_patient + '_' + filename
    shutil.copy(fromFilePathName, resultFilePathName)

    print('Copy to : ', resultFilePathName)

    count += 1      # 행 number

    return count

def moveFiles(count, id_patient):
    # 현재 폴더 경로의, dcm 파일들 리스트
    #dcm_list = os.listdir(path_cur)

    #for filename in dcm_list:

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
    #count = copyFiles(count, error_img_name)

    # 파일 이동 함수 호출
    count = moveFiles(count, error_img_name)

print('Num of copied file : ', count)