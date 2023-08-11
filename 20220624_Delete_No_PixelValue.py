# 2022.06.24
# - DCM 파일 확인해서, 0 외의 PixelValue 가 없을 경우, 파일 삭제 (or 이동)

from glob import glob
import copy
import pydicom
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path_dcm = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TestSET\WholeBrain_Vessel\dms_vessel'
path_copy = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TestSET\WholeBrain_Vessel\dms_vessel_noVal'

list_dcm = sorted(glob(path_dcm +'/*.dcm'))

cnt_no_pixel = 0

for path_cur_dcm in list_dcm:
    cur_dcm = pydicom.read_file(path_cur_dcm)
    cur_image = cur_dcm.pixel_array

    cur_image_min = int(cur_image.min())
    cur_image_max = int(cur_image.max())

    print('ID :', os.path.basename(path_cur_dcm))
    print('min : ', cur_image_min)
    print('max : ', cur_image_max)
    print('')

    if cur_image_min == 0 and cur_image_max == 0:
        shutil.move(path_cur_dcm, path_cur_dcm.replace('dms_vessel', 'dms_vessel_noVal'))
        cnt_no_pixel = cnt_no_pixel + 1

print('0 외의 Pixel 없는 데이터 개수 : ', cnt_no_pixel)