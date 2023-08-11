### 목적 ###
# -> csv에서, cHS Type별 Label을 확인하고,
# -> 각 Slice를 폴더별로 나눠서 복사해주기 위한 코드

import shutil
import os
import pandas as pd
from glob import glob

# 복사할 원본 파일 경로
path = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\0. all_slice_dcm'

# Label csv
df_label = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\Label_Slice_Wise.csv')

# 복사할 경로
resultPath_epidural = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\1. epidural'
resultPath_intraparenchymal = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\2. intraparenchymal'
resultPath_intraventricular = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\3. intraventricular'
resultPath_subarachnoid = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\4. subarachnoid'
resultPath_subdural = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220421_Small Hemorrhage Case\Small Hemorrhage 2차 데이터셋\5. subdural'

# 카운트
cnt_epidural = 0
cnt_intraparenchymal = 0
cnt_intraventricular = 0
cnt_subarachnoid = 0
cnt_subdural = 0

# 이름만 가지고 올 파일들의 리스트
list_dcm = sorted(glob(path + '/*.dcm'))

# 복사할 파일 리스트에서, 확장자 및 이름 변경
for cur_img in list_dcm:
    cur_name = os.path.basename(cur_img).replace('.dcm', '')

    cur_df = df_label[df_label['ID_Slice'] == cur_name]
    cur_df = cur_df.reset_index(drop=True)

    if cur_df['Label_epidural'][0]:
        shutil.copy(cur_img, os.path.join(resultPath_epidural, os.path.basename(cur_img)))
        print('copied to epidural : {}'.format(cur_name))
        cnt_epidural = cnt_epidural + 1

    if cur_df['Label_intraparenchymal'][0]:
        shutil.copy(cur_img, os.path.join(resultPath_intraparenchymal, os.path.basename(cur_img)))
        print('copied to intraparenchymal : {}'.format(cur_name))
        cnt_intraparenchymal = cnt_intraparenchymal + 1

    if cur_df['Label_intraventricular'][0]:
        shutil.copy(cur_img, os.path.join(resultPath_intraventricular, os.path.basename(cur_img)))
        print('copied to intraventricular : {}'.format(cur_name))
        cnt_intraventricular = cnt_intraventricular + 1

    if cur_df['Label_subarachnoid'][0]:
        shutil.copy(cur_img, os.path.join(resultPath_subarachnoid, os.path.basename(cur_img)))
        print('copied to subarachnoid : {}'.format(cur_name))
        cnt_subarachnoid = cnt_subarachnoid + 1

    if cur_df['Label_subdural'][0]:
        shutil.copy(cur_img, os.path.join(resultPath_subdural, os.path.basename(cur_img)))
        print('copied to subdural : {}'.format(cur_name))
        cnt_subdural = cnt_subdural + 1

print('all : ', cnt_epidural + cnt_intraparenchymal + cnt_intraventricular + cnt_subarachnoid + cnt_subdural)
print('cnt_epidural : ', cnt_epidural)
print('cnt_intraparenchymal : ', cnt_intraparenchymal)
print('cnt_intraventricular : ', cnt_intraventricular)
print('cnt_subarachnoid : ', cnt_subarachnoid)
print('cnt_subdural : ', cnt_subdural)
