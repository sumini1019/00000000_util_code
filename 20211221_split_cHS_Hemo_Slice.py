### 목적 ###
# -> cHS RSNA 데이터에 대해,
# -> 환자 별 dms 여부 결정

import pandas as pd
import random
import shutil
import os
#ID_12dfec0a3.dcm

is_train = True
if is_train:
    # 복사할 경로
    resultPath_hemo = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\train\hemo'
    resultPath_normal = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\train\normal'
    # resultPath_hemo = r'C:\20211222_임시폴더\train\hemo'
    # resultPath_normal = r'C:\20211222_임시폴더\train\normal'

    # 복사할 원본 파일 경로
    path = r'D:\00000000_Data\hemorrhage_classification\stage_2_train_images'

    # csv 데이터 프레임 로드
    df = pd.read_csv(r'D:\00000000_Data\hemorrhage_classification\rsna_train_binary.csv')
else:
    # 복사할 경로
    resultPath_hemo = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\test\hemo'
    resultPath_normal = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20211221_cHS_Slice기준_Classification\test\normal'

    # 복사할 원본 파일 경로
    path = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TestSet\stage_2_test_images'

    # csv 데이터 프레임 로드
    df = pd.read_csv(r'D:\00000000_Data\hemorrhage_classification\rsna_train_binary.csv')


# 인덱스 초기화
df.reset_index(drop=True, inplace=True)

# df = df[?:]
df.reset_index(drop=True, inplace=True)

# 복사 실패 카운팅
cnt_false = 0

# csv 파일 내, 행 개수만큼 반복
for i in range(0, len(df)):

    # 파일 이름 파싱
    name_file = df.loc[i].Image + ".dcm"

    # Hemo Slice 복사
    if df.loc[i]["hemorrhage"] == 1:
        try:
            shutil.copy(os.path.join(path, name_file), os.path.join(resultPath_hemo, name_file))
            # print("[Hemo] Succesee Copy : {}".format(name_file))
        except:
            print("[Hemo] False Copy : {}".format(name_file))
            cnt_false = cnt_false + 1
    # Normal Slice 복사
    else:
        try:
            shutil.copy(os.path.join(path, name_file), os.path.join(resultPath_normal, name_file))
            # print("[Normal] Succesee Copy : {}".format(name_file))
        except:
            print("[Normal] False Copy : {}".format(name_file))
            cnt_false = cnt_false + 1

print("Count_False : ", cnt_false)