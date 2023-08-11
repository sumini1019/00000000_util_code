# 2022.09.07
# - Series 데이터 모여있음
# - Series Label csv 기준으로, Hemo 여부에 따라 폴더 split

import pandas as pd
import shutil
import os
from glob import glob

# Json 뽑아올 root 폴더 (fold_0 ~ fold_10 반복 필요)
path_Source = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210310_cHS_Annotation_image_and_json\20220426_cHS_Annotation_Fold별 재구성 및 재평가\20220426_재평가 전 데이터'

# 데이터 복사할 타겟 경로
path_dest_Hemo = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\1. 2D Annotation\2. Data_Series (png, label)\Hemo'
path_dest_Normal = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\1. 2D Annotation\2. Data_Series (png, label)\Normal'

# Hemorrhage 정보 csv
df_label = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv')

# Source 폴더의 Sub 폴더 순회
list_SubFolder_Source = os.listdir(path_Source)[:11]

from knockknock import slack_sender
webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#noti", user_mentions=["@정수민"])
def main():
    # for cur_Source
    for cur_SubFolder in list_SubFolder_Source:
        # sub folder 경로
        path_cur_SubFolder = os.path.join(path_Source, cur_SubFolder)

        # 해당 폴더의 json 파일만 파싱
        list_json = glob(path_cur_SubFolder + '/*.json')

        # json 파일 순회
        for cur_json in list_json:
            # 현재 json 파일명
            cur_fn = os.path.basename(cur_json).replace('.json', '')
            # 데이터프레임에서, 해당 json이 Hemorrhage가 맞는지 확인
            cur_df = df_label[df_label['ID_Slice'] == cur_fn].reset_index(drop=True)

            # 정상인 경우
            if len(cur_df) == 1:
                # 데이터프레임에서, 해당 json이 Hemorrhage가 맞는지 확인
                if cur_df['Hemo_Slice'][0] == 1:

                    ################
                    # 복사할 때, 폴더 내에 index 정보 포함된 파일명으로 바꿔줘야 함
                    list_dest_file = os.listdir(os.path.join(path_dest_Hemo, cur_df['ID_Series'][0]))
                    # json 파일명
                    fn_json = os.path.basename(cur_json).replace('.json', '')


                    ##################################### 이 아래 부분 디버깅해서, 문제 없는지 체크 #####################
                    # 폴더 내에, json file과 동일한 아이디 있는지 확인
                    cnt_exist = 0
                    new_fn_json = ''
                    for cur_dest_file in list_dest_file:
                        if (fn_json in cur_dest_file) and ('.png' in cur_dest_file):
                            cnt_exist = cnt_exist + 1
                            new_fn_json = cur_dest_file.replace('.png', '.json')

                    # 동일한 아이디 있으면, 해당 아이디로 복사 (LabelMe 에서 보이도록)
                    if cnt_exist==1:
                        # 복사 (해당하는 Series 폴더로)
                        cur_path_source = cur_json
                        cur_path_dest = os.path.join(path_dest_Hemo, cur_df['ID_Series'][0], new_fn_json)
                        try:
                            shutil.copy(cur_path_source, cur_path_dest)
                        except:
                            print('Error - Copy 중, 에러 발생 ({})'.format(cur_fn))
                    else:
                        if cnt_exist==0:
                            print('Error - 해당 json 에 해당하는 png 파일이 없음! ({})'.format(fn_json))
                            continue
                        elif cnt_exist>1:
                            print('Error - 해당 json 에 해당하는 png 파일이 2개 이상! ({})'.format(fn_json))
                            continue

                else:
                    # 아니라면, 에러 발생
                    print('Error - Json이 Label 상, Hemorrhage가 아니라고 함 ({})'.format(cur_fn))
            # 에러 케이스
            else:
                # DataFrame에 정보가 없는 경우
                if len(cur_df) == 0:
                    print('Error - Label에 정보 없음 (Slice : {})'.format(cur_fn))
                    # continue
                # DataFrame에 정보가 2개 이상인 경우
                elif len(cur_df) > 1:
                    print('Error - Label에 정보가 2개 이상 (Slice : {})'.format(cur_fn))
                    # continue

main()