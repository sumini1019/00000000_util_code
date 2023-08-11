# 2022.09.07
# - Series 데이터 모여있음
# - Series Label csv 기준으로, Hemo 여부에 따라 폴더 split
# - ** Nifti 데이터에 대해서 진행

import pandas as pd
import shutil
import os

# Series ID 데이터 뽑아올 root 폴더
path_Source = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID_NIFTI'

# 데이터 복사할 타겟 경로
path_dest_Hemo = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\2. 3D Annotation\1. Data_Series\Hemo'
path_dest_Normal = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\2. 3D Annotation\1. Data_Series\Normal'

# Small Hemorrhage 정보 csv
df_label = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Patient_wise_ALL.csv')

# Hemorrhage - Series List
df_Hemo = df_label[df_label['Hemo_Patient']==1]
list_SeriesID_Hemo = df_Hemo['ID_Series']
# Normal - Series List
df_Normal = df_label[df_label['Hemo_Patient']==0]
list_SeriesID_Normal = df_Normal['ID_Series']

# 중복제거
list_SeriesID_Hemo = list(set(list_SeriesID_Hemo)) #집합set으로 변환
list_SeriesID_Normal = list(set(list_SeriesID_Normal)) #집합set으로 변환

from knockknock import slack_sender
webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#noti", user_mentions=["@정수민"])
def main():

    # 1. Hemo - SeriesID에 해당하는 폴더 복사
    # 카운트
    cnt_suc = 0
    cnt_fail = 0
    for cur_SeriesID in list_SeriesID_Hemo:
        try:
            shutil.copy(os.path.join(path_Source, cur_SeriesID + '.nii.gz'),
                        os.path.join(path_dest_Hemo, cur_SeriesID + '.nii.gz'))
            # print('copy success : {}'.format(cur_SeriesID))
            cnt_suc = cnt_suc + 1
        except:
            print('copy fail : {}'.format(cur_SeriesID))
            cnt_fail = cnt_fail + 1
    print('Hemo Num_Series : {}'.format(cnt_suc + cnt_fail))
    print('Hemo Num_Success : {}'.format(cnt_suc))
    print('Hemo Num_Fail : {}'.format(cnt_fail))


    # 2. Normal - SeriesID에 해당하는 폴더 복사
    # 카운트
    cnt_suc = 0
    cnt_fail = 0
    for cur_SeriesID in list_SeriesID_Normal:
        try:
            shutil.copy(os.path.join(path_Source, cur_SeriesID + '.nii.gz'),
                        os.path.join(path_dest_Normal, cur_SeriesID + '.nii.gz'))
            # print('copy success : {}'.format(cur_SeriesID))
            cnt_suc = cnt_suc + 1
        except:
            print('copy fail : {}'.format(cur_SeriesID))
            cnt_fail = cnt_fail + 1

    print('Normal Num_Series : {}'.format(cnt_suc + cnt_fail))
    print('Normal Num_Success : {}'.format(cnt_suc))
    print('Normal Num_Fail : {}'.format(cnt_fail))



main()