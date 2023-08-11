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
df_label = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID\cHS_RSNA_Label_Slice_wise_{}.csv'.format(mode))

# 이미지 위치
path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID'

# Patient-wise DataFrame 생성
cHS_RSNA_Label_Patient_wise = pd.DataFrame(columns=['ID_Series', 'Hemo_Series', 'Num_Slice', 'Num_Hemo_Slice'])

# Slice-wise DataFrame 생성
cHS_RSNA_Label_Slice_wise = pd.DataFrame(columns=['ID_Image', 'ID_Series', 'Hemo_Slice', 'Hemo_Series', 'Position', 'Seq'])


# 1. Series ID 기준으로, ID 리스트 뽑기
list_SeriesID = list(df_label.drop_duplicates(['ID_Series'])['ID_Series'])
# 2. Series ID 별로, Hemorrhage 누적해서 뽑기
for cur_SeriesID in list_SeriesID:
    # 현재 Series ID의 DataFrame
    cur_df = df_label[df_label['ID_Series'] == cur_SeriesID]

    # 현재 Series의 모든 Slice에 대해 반복
    for idx in cur_df.index:
        # Slice의 Path 설정
        path_cur_dicom = os.path.join(path_root, cur_df.loc[idx, 'ID_Series'], (cur_df.loc[idx, 'Image'] + '.dcm'))
        # Slice의 Dicom 파일 로드
        cur_dicom = pydicom.read_file(path_cur_dicom)
        # Position 파싱 후, DataFrame 에 삽입
        cur_df.loc[idx, 'position'] = cur_dicom[('0020', '0032')].value[2]

    # Position 기준으로, 재정렬 후 re-indexing
    cur_df = cur_df.sort_values(by=['position'], axis=0, ascending=0)
    cur_df = cur_df.reset_index(drop=True)

    # seq 순서 넣어주기 (정렬 순서대로)
    cur_df = cur_df.reset_index()
    cur_df.rename(columns={'index': 'seq'}, inplace=True)

    #################### Patient 기준 #############
    # 전체 슬라이스 개수
    num_slice = len(cur_df)
    # Hemorrhage 슬라이스 개수
    num_hemo_slice = cur_df['hemorrhage'].sum()
    # Patient-wise Hemo 여부
    is_hemo_patient_wise = 1 if num_hemo_slice >= 1 else 0

    # 3. Patient 기준, DataFrame 행 추가
    new_data = {'ID_Series': cur_SeriesID, 'Hemo_Series': is_hemo_patient_wise,
                'Num_Slice': num_slice, 'Num_Hemo_Slice': num_hemo_slice}
    cHS_RSNA_Label_Patient_wise = cHS_RSNA_Label_Patient_wise.append(new_data, ignore_index=True)
    ###############################################

    # Slice DataFrame에 Hemo 정보 넣어주기
    cur_df['Hemo_Series'] = is_hemo_patient_wise
    cur_df.rename(columns={'Image': 'ID_Image', 'ID_Series': 'ID_Series', 'hemorrhage': 'Hemo_Slice',
                           'position': 'Position', 'seq': 'Seq'}, inplace=True)
    cur_df.drop(columns=['normal'], inplace=True)

    # DataFrame 병합
    cHS_RSNA_Label_Slice_wise = cHS_RSNA_Label_Slice_wise.append(cur_df)

# 결과 index 초기화
cHS_RSNA_Label_Slice_wise.reset_index(drop=True)

# 4. 결과 CSV 저장
cHS_RSNA_Label_Patient_wise.to_csv('cHS_RSNA_Label_Patient_wise_{}.csv'.format(mode), index=False)
cHS_RSNA_Label_Slice_wise.to_csv('cHS_RSNA_Label_Slice_wise_new_{}.csv'.format(mode), index=False)


