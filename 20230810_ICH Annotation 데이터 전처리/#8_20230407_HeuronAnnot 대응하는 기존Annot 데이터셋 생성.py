import shutil
from module_sumin.utils_sumin import read_csv_autodetect_encoding
import os
import pandas as pd
import glob


# 2023.04.07
# - Heuron Annot에 대응하는 기존 Annot 데이터셋을 생성해야함
path_src_PrevAnnot = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_for_Segmentation★★★'
path_dst_PrevAnnot = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\6_until_7th(Prev_Annot)'
# path_dst_PrevAnnot = r'D:\00000000_Data\20230403_HeuronAnnotation\3_png(Prev_Annot)'

os.makedirs(os.path.join(path_dst_PrevAnnot, 'image_png'), exist_ok=True)
os.makedirs(os.path.join(path_dst_PrevAnnot, 'label_binary'), exist_ok=True)
os.makedirs(os.path.join(path_dst_PrevAnnot, 'label_subtype'), exist_ok=True)

# 1. Heuron Annot의 Slice 기준 csv 사용해서, 대상 Series 가지고 오기
# df_HeuronAnnot = read_csv_autodetect_encoding(
#     r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230406_GT_ICH_Annotation_Slicewise_n73902_5차수령데이터까지.csv')
# df_HeuronAnnot = read_csv_autodetect_encoding(
#     r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_GT_ICH_Annotation_Slicewise_n105921_6차수령데이터까지.csv')
df_HeuronAnnot = read_csv_autodetect_encoding(
    r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230811_GT_ICH_Annotation_Slicewise_n124062_7차수령데이터까지.csv')


list_Series = list(set(df_HeuronAnnot['Series_ID']))

# 2. 기존 Annot의 csv 사용해서, ID_Series에 해당하는 모든 ID_Slice의 이미지와, label 복사하기
df_PrevAnnot = read_csv_autodetect_encoding(
    r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv')

# Get list of all files in the directories
src_images = set(glob.glob(os.path.join(path_src_PrevAnnot, 'image_png', '*.png')))
src_labels_binary = set(glob.glob(os.path.join(path_src_PrevAnnot, 'label_binary', '*.png')))
src_labels_subtype = set(glob.glob(os.path.join(path_src_PrevAnnot, 'label_subtype', '*.png')))

cnt = 0
for index, cur_Series in enumerate(list_Series):
    print(f'Processing index: {index} / {len(list_Series)-1}', end='\r')

    # 현재 Series에 해당하는 Slice ID 리스트
    list_Slice = list(df_PrevAnnot[df_PrevAnnot['ID_Series'] == cur_Series[5:]]['ID_Slice'])

    # Slice에 해당하는 image / label 복사
    for cur_Slice in list_Slice:
        # print(f'Cnt : {cnt}\n{cur_Slice}')

        # 1. image
        path_src_img = os.path.join(path_src_PrevAnnot, 'image_png', cur_Slice + '_img.png')
        path_dest_img = os.path.join(path_dst_PrevAnnot, 'image_png', cur_Slice + '_img.png')
        if path_src_img in src_images:
            shutil.copy(path_src_img, path_dest_img)

        # 2. Label (binary)
        path_src_label_binary = os.path.join(path_src_PrevAnnot, 'label_binary', cur_Slice + '_label.png')
        path_dest_label_binary = os.path.join(path_dst_PrevAnnot, 'label_binary', cur_Slice + '_label.png')
        if path_src_label_binary in src_labels_binary:
            shutil.copy(path_src_label_binary, path_dest_label_binary)

        # 3. Label (subtype)
        path_src_label_subtype = os.path.join(path_src_PrevAnnot, 'label_subtype', cur_Slice + '_label.png')
        path_dest_label_subtype = os.path.join(path_dst_PrevAnnot, 'label_subtype', cur_Slice + '_label.png')
        if path_src_label_subtype in src_labels_subtype:
            shutil.copy(path_src_label_subtype, path_dest_label_subtype)

        cnt += 1
