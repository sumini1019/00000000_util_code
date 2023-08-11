# 2023.02.08
# - .nrrd 파일 -> .seg.nrrd 파일로 변경

import pandas as pd
import shutil
import os
from glob import glob

path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230127_1차 수령 데이터에 대한 검토 대상 데이터\데이터'

list_nrrd = glob(path_root + '/*.nrrd')

for idx, item in enumerate(list_nrrd):
    # 파일명
    fn = os.path.basename(item)
    # 확장자 변경
    fn_new = fn.replace('.nrrd', '.seg.nrrd')
    # 파일명 변경
    os.rename(os.path.join(path_root, fn), os.path.join(path_root, fn_new))