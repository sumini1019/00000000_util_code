# 2023.03.08
# - ICH Annotation이 폴더 별 정리되어 있음
    # - Subtype_Grade 형태
# - Subtype 기준 및 각 폴더의 파일 개수를 표로 생성하는 코드


import os
import csv
import matplotlib.pyplot as plt
import pydicom
import glob
import SimpleITK as sitk
import numpy as np
from module_sumin.utils_sumin import read_csv_autodetect_encoding

# 폴더 경로 설정
path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID'
path_df = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Patient_wise_ALL (Subtype 정보 포함).csv'

os.scandir(path_root)
list_folder = os.listdir(path_root)

df = read_csv_autodetect_encoding(path_df)

for idx, cur_folder in enumerate(list_folder):
    list_dicom = glob.glob(os.path.join(path_root, cur_folder) + '/*.dcm')

    sample_dcm = pydicom.dcmread(list_dicom[2])

    # Slice Thickness 정보 확인
    try:
        if hasattr(sample_dcm, "SliceThickness"):
            slice_thickness = sample_dcm.SliceThickness
            print(f"Slice Thickness: {slice_thickness}")
        # Slice Thickness 정보 없으면, 앞-뒤 슬라이스의 Image Position으로 계산
        else:
            # DICOM 시리즈 ID 가져오기
            series_ID = sitk.ImageSeriesReader.GetGDCMSeriesIDs(os.path.join(path_root, cur_folder))
            # 시리즈의 파일 이름 가져오기 (정렬됨)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(os.path.join(path_root, cur_folder),
                                                                              series_ID[0])

            # 앞-뒤 DICOM 파일의 Image Position 가져오기
            dcm1 = sitk.ReadImage(series_file_names[0])
            dcm2 = sitk.ReadImage(series_file_names[1])
            pos1 = np.array(dcm1.GetOrigin())
            pos2 = np.array(dcm2.GetOrigin())

            # 두 슬라이스 사이의 거리 계산
            slice_thickness = np.linalg.norm(pos2 - pos1)
            slice_thickness = round(slice_thickness, 2)

        df.loc[df['ID_Series'] == cur_folder, 'Slice_Thickness'] = slice_thickness
        print(f"ID : {cur_folder} :: Computed Slice Thickness: {slice_thickness}")


    except KeyError:
        print(f'Error - ID : {cur_folder} :: Slice Thickness 계산 불가')

    if idx % 100 == 0:
        df.to_csv('cHS_RSNA_Label_Patient_wise_ALL (Subtype 정보 포함).csv')

df.to_csv('cHS_RSNA_Label_Patient_wise_ALL (Subtype 정보 포함).csv')