### 목적 ###
# -> csv를 통해, 환자 ID 확인 후,
# -> 폴더 나눠서 복사



import shutil
import os
import pandas as pd
from glob import glob


# 복사할 원본 파일 경로
# path = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Half/fold1'
path_data = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20210914_DMS분류_LDCT_Norm_아주대_길병원일부\Test_LVO_환자기준\새 폴더'
path_cur = ''
# 복사할 경로 - Case #1 (DMS)
# resultPath = 'Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/Half/fold1/dms'
resultPath = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20210914_DMS분류_LDCT_Norm_아주대_길병원일부\Test_LVO_환자기준'

list_image = sorted(glob(path_data + '/*.dcm'))
#
# # csv 데이터 프레임 로드
# df = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/DATA_Annotation_list_csv.csv')

cnt_image = 0

for cur_image in list_image:
    cur_image_base = os.path.basename(cur_image)
    id_image = cur_image_base[:3]

    if not os.path.isdir(os.path.join(resultPath, id_image)):
        os.mkdir(os.path.join(resultPath, id_image))

        os.mkdir(os.path.join(resultPath, id_image, 'ELVO'))
        os.mkdir(os.path.join(resultPath, id_image, 'ELVO', 'LH'))
        os.mkdir(os.path.join(resultPath, id_image, 'ELVO', 'RH'))

    if '_L.dcm' in cur_image_base:
        shutil.copy(os.path.join(path_data, cur_image_base), os.path.join(resultPath, id_image, 'ELVO', 'LH',cur_image_base))
    elif '_R.dcm' in cur_image_base:
        shutil.copy(os.path.join(path_data, cur_image_base),
                    os.path.join(resultPath, id_image, 'ELVO', 'RH', cur_image_base))
    else:
        print('########### File 방향 정보 없음')



    cnt_image = cnt_image + 1

print(cnt_image)
