# 2022.06.16
# - Fold별 데이터에 대한 DataFrame 만들어오기
# -

import pandas as pd
import shutil
import os
from glob import glob

# Hemo 데이터 Fold 폴더 경로
path_target = r'H:\20220614_이어명 인턴 전달\1. cHS_Annotation_DataSet\2. Fold별 DataSet'
# Series ID 데이터를 복사해 올 폴더 (SeiresID 별로 폴더 별개)
path_source = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID★★★'
# 복사 할 경로
path_dest = r'H:\20220614_이어명 인턴 전달\1. cHS_Annotation_DataSet\1. Dicom 원본 (Series기준으로, Fold별로)\fold9'
# Slice-wise Label 데이터프레임
path_label_csv = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv'

# Slice 리스트
list_fold = os.listdir(path_target)

# Label DataFrame
df_label = pd.read_csv(path_label_csv)

# Hemorrhage 만 존재하는 Dataframe 파싱
df_label_hemo = df_label[df_label['Hemo_Slice']==1].reset_index(drop=True)
# 열 추가
df_label_hemo['Fold'] = ""

cnt_err = 0
cnt_suc = 0

# 폴더별로, 순회하면서 Hemo slice 파일명 읽어들이고, Dataframe에 폴드 정보 저장
for cur_fold in list_fold:
    cur_slice_list = glob(os.path.join(path_target, cur_fold) + '/*.png')

    for cur_slice_path in cur_slice_list:
        cur_slice_fn = os.path.basename(cur_slice_path).replace(".png", "")

        cur_df = df_label_hemo[df_label_hemo['ID_Slice'] == cur_slice_fn].reset_index(drop=True)
        if len(cur_df) == 0:
            print('Error - {} 파일은, Label csv에 없음'.format(cur_slice_fn))
            cnt_err = cnt_err + 1
            continue
        else:
            idx = int(df_label_hemo[df_label_hemo['ID_Slice'] == cur_slice_fn].index[0])
            df_label_hemo.loc[idx, 'Fold'] = cur_fold
            cnt_suc = cnt_suc + 1

    df_label_hemo.to_csv('cHS_RSNA_Label_Slice_wise_forSegmentation_onlyHemorrhage_Fold.csv', index=False)

df_label_hemo.to_csv('cHS_RSNA_Label_Slice_wise_forSegmentation_onlyHemorrhage_Fold.csv', index=False)





