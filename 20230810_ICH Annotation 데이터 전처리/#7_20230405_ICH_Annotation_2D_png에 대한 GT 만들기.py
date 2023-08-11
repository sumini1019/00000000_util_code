# 2023.04.06
# - 3d 데이터를, 2d 데이터로 변환하면서,
# - 각 slice 기준의 GT는 만들지 않았음
# - 방법 상세
#   1. png 폴더에서 파일 리스트 뽑기
#   2. Series GT 기준으로, ID 확인해서 ID에 해당하는 type (train, val, test) 과 동일한 type으로 slice GT 생성
#   3. Slice GT 생성 시, Slice별 Column 종류는 아래와 같음
#       - [Image	Label	Type	P_Voxel_Num_EDH	Label_EDH	P_Voxel_Num_ICH	Label_ICH	P_Voxel_Num_IVH	Label_IVH
#          P_Voxel_Num_SAH	Label_SAH	P_Voxel_Num_SDH	Label_SDH	P_Voxel_Num_SDH(Chronic)	Label_SDH(Chronic)
#          P_Voxel_Num_HemorrhagicContusion	Label_HemorrhagicContusion]

import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
import cv2
import nrrd
import pandas as pd
from module_sumin.utils_sumin import read_csv_autodetect_encoding

columns = ['Image', 'Label', 'Series_ID', 'Type', 'P_Voxel_Num_EDH', 'Label_EDH', 'P_Voxel_Num_ICH', 'Label_ICH', 'P_Voxel_Num_IVH',
           'Label_IVH', 'P_Voxel_Num_SAH', 'Label_SAH', 'P_Voxel_Num_SDH', 'Label_SDH', 'P_Voxel_Num_SDH(Chronic)',
           'Label_SDH(Chronic)', 'P_Voxel_Num_HemorrhagicContusion', 'Label_HemorrhagicContusion']

selected_classes = ['BG', 'EDH', 'ICH', 'IVH', 'SAH', 'SDH', 'SDH(Chronic)', 'HemorrhagicContusion']
class_dict = {'EDH': 1, 'ICH': 2, 'IVH': 3, 'SAH': 4, 'SDH': 5, 'SDH(Chronic)': 6, 'HemorrhagicContusion': 7}

df = pd.DataFrame(columns=columns)

# path_GT_csv = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230406_GT_ICH_Annotation_n2232_5차수령데이터까지.csv'
# path_slice = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\image_png'
# path_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\label_png_subtype'

# path_GT_csv = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_GT_ICH_Annotation_n3183_6차수령데이터까지.csv'
# path_slice = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\image_png'
# path_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\label_png_subtype'

path_GT_csv = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230811_GT_ICH_Annotation_n3727_7차수령데이터까지.csv'
path_slice = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\image_png'
path_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\label_png_subtype'



df_GT = read_csv_autodetect_encoding(path_GT_csv)

list_slice = os.listdir(path_slice)

for index, item in enumerate(list_slice):
    print(f'Processing index: {index} / {len(list_slice)-1} || FileName: {item}', end='\r')

    series_ID = item.split('_idx')[0]
    fn_series_img = series_ID + '.nii.gz'
    cur_type = df_GT.loc[df_GT['Image'] == fn_series_img, 'Type'].values[0]

    # 이에 대응하는 label 파일
    fn_label = item.replace('_idx', '-label_idx')
    path_label = os.path.join(path_label_subtype, fn_label)

    label_image = Image.open(path_label)
    label_array = np.array(label_image)

    unique, counts = np.unique(label_array, return_counts=True)

    pixel_dict = {}

    pixel_dict = {}
    for i, class_name in enumerate(selected_classes):
        if class_name == 'BG':
            continue
        label_val = i
        pixel_val_counts = np.sum(label_array == i)
        if pixel_val_counts >= 10:
            label_name = f'Label_{class_name}'
            pvoxel_name = f'P_Voxel_Num_{class_name}'
            pixel_dict[pvoxel_name] = pixel_val_counts
            pixel_dict[label_name] = 1
        else:
            label_name = f'Label_{class_name}'
            pvoxel_name = f'P_Voxel_Num_{class_name}'
            pixel_dict[pvoxel_name] = pixel_val_counts
            pixel_dict[label_name] = 0

    # 결과 dictionary 최종
    result_dict = {'Image': item, 'Label': fn_label, 'Series_ID': series_ID, 'Type': cur_type}
    result_dict.update(pixel_dict)

    df = pd.concat([df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)

path_save = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령'
df.to_csv(os.path.join(path_save, f'20230811_GT_ICH_Annotation_Slicewise_n{len(df)}_7차수령데이터까지.csv'), index=False)