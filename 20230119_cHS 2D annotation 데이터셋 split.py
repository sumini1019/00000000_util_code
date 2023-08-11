### 목적 ###
# -> Series ID 별로 부여된, csv 를 기반으로 함
# -> Series ID에 해당하는 Slice ID를 찾아서 Dataframe merge 후 저장

import pandas as pd
import random
import shutil
import os

list_dataset = ['train', 'val', 'test']
list_df = []
for cur_dataset in list_dataset:
    # Series ID 기준 csv
    path_csv_original = r'D:\OneDrive\00000000_Code\20230109_cHS_3d_classification\20230110_ICH_{}.csv'.format(cur_dataset)
    # Slice ID 기준 Ground Truth csv
    path_csv_slice_wise = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv'
    # 분류 후 저장할 csv 위치
    path_csv_dest = r'D:\OneDrive\00000000_Code\20230109_cHS_3d_classification\20230119_ICH_Slice_wise_{}.csv'.format(cur_dataset)

    df_original = pd.read_csv(path_csv_original)
    df_slice_wise = pd.read_csv(path_csv_slice_wise)

    df_original = df_original.rename(columns={'Hemo_Patient': 'Hemo_Series'})
    df_original = df_original.drop('Num_Slice', axis=1)

    # ID_Series 기준으로 합성
    merged_df = pd.merge(df_original, df_slice_wise, on='ID_Series', how='inner')

    merged_df.to_csv(path_csv_dest, index=False)

    list_df.append(merged_df)

print(len(list_df[0])+len(list_df[1])+len(list_df[2]))