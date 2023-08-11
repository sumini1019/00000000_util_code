# 2023.02.08
# - RSNA 데이터셋에 특정 dcm이 로드가 안되는 문제 있는 것으로 의심 됨
#   - ValueError: The length of the pixel data in the dataset (153710 bytes) doesn't match the expected length (524288 bytes). The dataset may be corrupted or there may be an issue with the pixel data handler.
# - pixel array 한개씩 꺼내서, 로드해보는 방식으로 확인

import pandas as pd
import shutil
import os
from glob import glob
import nrrd
import pydicom

data_path_NoSplit = r'D:\20230130_RSNA 데이터'

list_dcm = os.listdir(data_path_NoSplit)

for idx, item in enumerate(list_dcm):
    path_dcm = os.path.join(data_path_NoSplit, item)

    if idx%10000 == 0:
        print(f'idx : {idx}')
    try:
        dataset = pydicom.read_file(path_dcm)
        dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image = dataset.pixel_array
    except:
        print(f'{item} 이미지 로드 실패 : {path_dcm}')
