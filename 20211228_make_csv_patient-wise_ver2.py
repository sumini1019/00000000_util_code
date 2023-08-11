### 목적 ###
# -> Slice 별, Label CSV를 읽고,
# -> Patient 별, Label CSV 만들기

import pandas as pd
import random
import shutil
import os
from glob import glob
import pydicom

# mode
mode = 'TRAIN'

# Hemo Label 정보
df_label = pd.read_csv(r'C:\Users\user\Downloads\DataSET_Annot_v211022 (version 1).csv')

# # 이미지 위치
# path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID'
#
# Patient-wise DataFrame 생성
df_Patient_wise = pd.read_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\LVO_GT_WholeBrain.csv')
df_Patient_wise = df_Patient_wise.assign(DMS_WB=0, DMS_R=0, DMS_L=0, EIC_WB=0, EIC_R=0, EIC_L=0)

#
# # Slice-wise DataFrame 생성
# cHS_RSNA_Label_Slice_wise = pd.DataFrame(columns=['ID_Image', 'ID_Series', 'Hemo_Slice', 'Hemo_Series', 'Position', 'Seq'])


# 1. Series ID 기준으로, ID 리스트 뽑기
list_SeriesID = list(df_label.drop_duplicates(['ID'])['ID'])
# 2. Series ID 별로, Hemorrhage 누적해서 뽑기
for cur_SeriesID in list_SeriesID:
    cur_df = df_label[df_label['ID'] == cur_SeriesID]

    # 현재 ID의, DMS L/R, EIC L/R 에 대해 counting
    DMS_RH = 1 if sum(cur_df['DMS Eval. - RH']) > 0 else 0
    DMS_LH = 1 if sum(cur_df['DMS Eval. - LH']) > 0 else 0
    EIC_RH = 1 if sum(cur_df['EIC Eval. - RH']) > 0 else 0
    EIC_LH = 1 if sum(cur_df['EIC Eval. - LH']) > 0 else 0
    # DMS / EIC 전체 (whole brain)에 대한 annotation
    DMS_WB = 1 if DMS_RH or DMS_LH else 0
    EIC_WB = 1 if EIC_RH or EIC_LH else 0

    # Patient-wise DataFrame 수정
    df_Patient_wise.loc[df_Patient_wise['ID'] == cur_SeriesID, 'DMS_R'] = DMS_RH
    df_Patient_wise.loc[df_Patient_wise['ID'] == cur_SeriesID, 'DMS_L'] = DMS_LH
    df_Patient_wise.loc[df_Patient_wise['ID'] == cur_SeriesID, 'EIC_R'] = EIC_RH
    df_Patient_wise.loc[df_Patient_wise['ID'] == cur_SeriesID, 'EIC_L'] = EIC_LH
    df_Patient_wise.loc[df_Patient_wise['ID'] == cur_SeriesID, 'DMS_WB'] = DMS_WB
    df_Patient_wise.loc[df_Patient_wise['ID'] == cur_SeriesID, 'EIC_WB'] = EIC_WB

# 결과 index 초기화
df_Patient_wise.reset_index(drop=True)

# 4. 결과 CSV 저장
df_Patient_wise.to_csv('cELVO_Label_Patient_wise.csv', index=False)