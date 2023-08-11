
### 목적 ###
# -> 여러 환자의 폴더에서, NCCT 폴더의 데이터만, 한개 폴더로 모으는 작업

import pandas as pd
import random
import shutil
import os
from glob import glob

# path_source = r'Z:\Stroke\TestData\cELVO_Prep_DataSET\Whole_Brain_SET\TrainSET'
path_source = r'Z:\Stroke\TestData\cELVO_Prep_DataSET\Whole_Brain_SET\TestSET'

list_folder = os.listdir(path_source)

path_result = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20211005_DMS분류_WholeBrain_아주대_길병원'

df = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20211005_DMS분류_WholeBrain_아주대_길병원\TestSET_Annot.csv')

list_image_used_for_dms = list(df[df['USED for DMS - RH']==1]['ID_Filename'])

df_used_for_dms = df[df['USED for DMS - RH']==1]
df_used_for_dms.reset_index(drop=True, inplace=True)

for cur_id in list_folder:
    cur_path = os.path.join(path_source, cur_id, 'ELVO', 'Whole')

    cur_list_file = glob(cur_path + '/*.dcm')

    for cur_file in cur_list_file:
        # USED for DMS가 0이라면, 복사
        if os.path.basename(cur_file).replace('_WB.dcm', '.dcm') in list_image_used_for_dms:

            # dms / normal 검사
            is_dms = int(df_used_for_dms[
                    df_used_for_dms['ID_Filename'] == os.path.basename(cur_file).replace('_WB.dcm', '.dcm')]['DMS'])

            if is_dms:
                # dms 폴더에 복사
                # shutil.copyfile(cur_file, os.path.join(path_result, 'dms', os.path.basename(cur_file)))
                shutil.copyfile(cur_file, os.path.join(path_result, 'dms(for_test)', os.path.basename(cur_file)))
            else:
                # normal 폴더에 복사
                shutil.copyfile(cur_file, os.path.join(path_result, 'normal(for_test)', os.path.basename(cur_file)))






