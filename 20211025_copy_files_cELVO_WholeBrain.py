
### 목적 ###
# -> 여러 환자의 폴더에서, NCCT 폴더의 데이터만, 한개 폴더로 모으는 작업

import pandas as pd
import random
import shutil
import os
from glob import glob

# path_source = r'Z:\Stroke\TestData\cELVO_Prep_DataSET\Whole_Brain_SET_v211022\TrainSET_v211022'
path_source = r'Z:\Stroke\TestData\cELVO_Prep_DataSET\Whole_Brain_SET_v211022\TestSET_v211022'

list_folder = os.listdir(path_source)

# path_result = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20211025_DMS분류_WholeBrain_아주대_길병원_파일이름수정'
path_result = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20211025_DMS분류_Hemi_아주대_길병원_파일이름수정\RH'

# df = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20211025_DMS분류_WholeBrain_아주대_길병원_파일이름수정\TrainSET_Annot_v211022.csv')
df = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20211025_DMS분류_WholeBrain_아주대_길병원_파일이름수정\TestSET_Annot_v211022.csv')

list_image_used_for_dms = list(df[df['USED for DMS - RH']==1]['ID_Filename'])

df_used_for_dms = df[df['USED for DMS - RH']==1]
df_used_for_dms.reset_index(drop=True, inplace=True)

copy_type = 'RH' #'LH'  'Whole'

for cur_id in list_folder:

    cur_path = os.path.join(path_source, cur_id, 'ELVO/{}'.format(copy_type))

    cur_list_file = glob(cur_path + '/*.dcm')

    for cur_file in cur_list_file:
        file_type = copy_type
        if copy_type=='Whole':
            file_type = 'WB'
        # USED for DMS가 0이라면, 복사
        # if os.path.basename(cur_file).replace('_LH.dcm', '.dcm') in list_image_used_for_dms:
        if os.path.basename(cur_file).replace('_{}.dcm'.format(file_type), '.dcm') in list_image_used_for_dms:
        # if os.path.basename(cur_file).replace('_WB.dcm', '.dcm') in list_image_used_for_dms:

            # dms / normal 검사
            # is_dms = int(df_used_for_dms[df_used_for_dms['ID_Filename'] == os.path.basename(cur_file).replace('_LH.dcm', '.dcm')]['DMS Eval. - LH'])
            is_dms = int(df_used_for_dms[df_used_for_dms['ID_Filename'] == os.path.basename(cur_file).replace('_RH.dcm', '.dcm')]['DMS Eval. - RH'])
            # is_dms = int(df_used_for_dms[df_used_for_dms['ID_Filename'] == os.path.basename(cur_file).replace('_WB.dcm', '.dcm')]['DMS Eval. - RH'])\
            #          + int(df_used_for_dms[df_used_for_dms['ID_Filename'] == os.path.basename(cur_file).replace('_WB.dcm', '.dcm')]['DMS Eval. - LH'])

            if is_dms:
                # dms 폴더에 복사
                # shutil.copyfile(cur_file, os.path.join(path_result, 'dms', os.path.basename(cur_file)))
                shutil.copyfile(cur_file, os.path.join(path_result, 'dms(for_test)', os.path.basename(cur_file)))
            else:
                # normal 폴더에 복사
                # shutil.copyfile(cur_file, os.path.join(path_result, 'normal', os.path.basename(cur_file)))
                shutil.copyfile(cur_file, os.path.join(path_result, 'normal(for_test)', os.path.basename(cur_file)))






