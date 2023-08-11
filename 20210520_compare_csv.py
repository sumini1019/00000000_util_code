### 목적 ###
# csv 2개 읽어들이고, 비교하는 작업
# 1. 한쪽 csv에만 존재하는 데이터 리스트 뽑기
# 2. DMS / EIC Label이 변경된 데이터 리스트 뽑기

import shutil
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv


# 비교할 2개 csv 로드
df_before = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/DATA_Annotation_list_csv.csv')
df_after = pd.read_csv('Z:/Sumin_Jung/00000000_DATA/3_cELVO/20210507_DMS_김도현책임님 전달 데이터/Dense_MCA_Sign/DATA_Annotation_list_dms_v210518.csv')

print('s')

d1 = df_before['ID_Filename']
d2 = df_after['ID_Filename']

d1 = list(d1)
d2 = list(d2)

# 없어진 파일 리스트
list_file_deleted = []
# DMS가 바뀐 리스트
list_file_label_modified = []

for cur_id in d1:
    if not cur_id in d2:
        print('dont have', cur_id)
        list_file_deleted.append(cur_id)
    else:
        print('have ', cur_id)

list_df = pd.DataFrame(list_file_deleted)

#list_df.to_csv('deleted_dms_list.csv')



# after csv에서, DMS_R이나, DMS_L이 기존과 바뀐 ID_Filename 리스트 뽑아보기
    # 1. after_csv 의 행 하나하나 돌기
for index, row in df_after.iterrows():
    # 2. Before&After 에서, 해당 Filename의, DMS_R 이나 DMS_L 값이 다른 Filename 파싱
    cur_name = row['ID_Filename']   # 현재 Filename 파싱
    # before 에서, 해당 filename의 행 찾아서 갖고 오기
    before_row = df_before[df_before['ID_Filename'] == cur_name]

    # 변경된 데이터의 row
    after_row = (pd.DataFrame(row)).transpose()

    # 동일한지 검사
    # bool_DMS_R = after_row.loc[index, 'DMS - R'] == before_row.loc[index, 'DMS_R']
    # bool_DMS_L = after_row.loc[index, 'DMS - L'] == before_row.loc[index, 'DMS_L']



    bool_DMS_R = after_row.DMS_R.values == before_row.DMS_R.values
    bool_DMS_L = after_row.DMS_L.values == before_row.DMS_L.values

    if bool_DMS_R[0]==False:
        print('R False')
        if bool_DMS_L[0]==False:
            print('Both False')

    # 동일하지 않다면,
    if ((bool_DMS_R[0]==False) or (bool_DMS_L[0]==False)):
        # 3. 해당 Filename 리스트에 넣기
        list_file_label_modified.append(cur_name)

list_df2 = pd.DataFrame(list_file_label_modified)

list_df2.to_csv('list_file_label_modified.csv')

    # 4. 해당하는 Filename 들만, 각 폴드에서 직접 하나한 옮겨주기