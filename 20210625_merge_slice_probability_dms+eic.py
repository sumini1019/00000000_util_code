### 목적 ###
# -> DMS, EIC Probability를 저장한 csv 2개를 조합하는 코드

import pandas as pd
import random
import shutil
import os

################### DMS 파일들만, fold별 복사

# csv 경로
result_csv_dms = 'D:/00000000 Code/20200728 Semantic Segmentation_pytorch official/pytorch_segmentation-master/Result_Label_and_Prob.csv'
result_csv_eic = 'D:/00000000 Code/20200728 Semantic Segmentation_pytorch official/pytorch_segmentation-master/Result_eic.csv'

# csv 데이터 프레임 로드
df_dms = pd.read_csv(result_csv_dms)
df_eic = pd.read_csv(result_csv_eic)

list_dms_patient = list(set(df_dms['ID']))

for i in range(0, len(df_eic)):
    if not df_eic.loc[i].ID in list_dms_patient:
        print('{} is not found in list'.format(df_eic.loc[i].ID))
        df_eic = df_eic.drop(i)

df_eic.reset_index(drop=True, inplace=True)

df_dms = df_dms[['ID_Filename', 'Pred_DMS_R_Quarter', 'Pred_DMS_L_Quarter']]

df_merged = df_eic.merge(df_dms, on = 'ID_Filename', how='outer')

df_merged.to_csv('merged_dms_eic.csv')
