### 목적 ###
# -> 폴더 내에서, 랜덤한 폴더를 선택하고,
# -> 해당 폴더만 복사

import pandas as pd
import random
import shutil
import os

# 복사할 경로
resultPath = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\test\All_SeriedID 기준'
resultPath_hemo = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\test\hemo'
resultPath_normal = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\test\normal'

# 복사할 원본 파일 경로
path = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID'

# 폴더 리스트
list_folder = os.listdir(path)

# csv 데이터 프레임 로드
df = pd.read_csv(r'D:\00000000_Data\hemorrhage_classification\rsna_train_binary.csv')

# 복사할 폴더 리스트
list_sample = random.sample(list_folder, 150)
# 랜덤하게 폴더 뽑기
for cur_folder in list_sample:
    # 폴더 내, Hemo / Normal Slice 복사
    list_file = os.listdir(os.path.join(path, cur_folder))

    # 환자가 hemo인가?
    is_hemo = False

    # 파일 별로 Hemo/Normal 확인
    for cur_file in list_file:
        name_file = cur_file.replace('.dcm', '')

        # Hemorrhage Slice인 경우
        if int(df[df['Image'] == name_file]['hemorrhage']):
            shutil.copy(os.path.join(path, cur_folder, cur_file), os.path.join(resultPath_hemo, cur_file))
            # Hemo Slice 있으므로, Hemo Series
            is_hemo = True
        # Normal Slice인 경우
        else:
            shutil.copy(os.path.join(path, cur_folder, cur_file), os.path.join(resultPath_normal, cur_file))

    # 환자의 Hemorrhage 여부에 따라, 폴더 복사
    if is_hemo:
        shutil.copytree(os.path.join(path, cur_folder),
                        os.path.join(resultPath, 'hemo series', cur_folder))
    else:
        shutil.copytree(os.path.join(path, cur_folder),
                        os.path.join(resultPath, 'normal series', cur_folder))


df_sample = pd.DataFrame(list_sample, columns=['SeriesID'])
df_sample.to_csv('List_sample.csv')