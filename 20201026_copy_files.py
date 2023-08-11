### 목적 ###
# -> cHS에서, json 에러난 이미지들만 뽑으려고 만듦

# 폴더의 파일 리스트 뽑고,
# 파일 이름이 같지만, 확장자가 다른 파일들을 모두 복사하는 코드



import shutil
import os
import pandas as pd

# 원본 파일 경로
#path = 'D:/cHS_Test/image'
path = '//heuronnas/연구개발부/Sumin_Jung/20201022_backup/20201021_cHS_Annotated/홍나래 (3,000)/image'
path_cur = ''
# 복사할 경로
#resultPath = 'D:/cHS_Test/image_error'
resultPath = '//heuronnas/연구개발부/Sumin_Jung/20201022_backup/20201021_cHS_Annotated/홍나래 (3,000)/image_error'
# json 파일 경로
jsonPath = '//heuronnas/연구개발부/Sumin_Jung/20201022_backup/20201021_cHS_Annotated/홍나래 (3,000)/json_2point'

# error 발생한 json 파일 리스트
error_img_list = os.listdir(jsonPath)

# json 파일 이름과 동일한 image 리스트 생성
for i in range(len(error_img_list)):
    error_img_list[i] = os.path.splitext(error_img_list[i])[0] + '.png'

count = 0

# csv 데이터 프레임 저장
#df = pd.read_csv('C:/00000000 Data/20200327 ICH Data/Top3/input/rsna_train (exclude intact).csv')
df = pd.DataFrame(error_img_list)

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
    error_img_name = df.loc[i-1][0]
    # id_patient = filename_cur + '.dcm'

    # 파일 복사 함수 호출
    count = moveFiles(count, error_img_name)

print('Num of copied file : ', count)