# 2023.08.10
# - 수령한 데이터에서
# - nrrd 데이터만, 특정 폴더로 복사
# - 2d segmentation 용 데이터 생성 목적

import os
import shutil

# 소스 폴더 경로들
src_paths = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_merged_dataset_until_8th'

# 대상 폴더 경로
dst_path = r"D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_8th)"
os.makedirs(dst_path, exist_ok=True)

# a 폴더에서 .nrrd 파일만 선택하여 b 폴더로 복사
for filename in os.listdir(src_paths):
    if filename.endswith(".nrrd"):
        source_path = os.path.join(src_paths, filename)
        dest_path = os.path.join(dst_path, filename)

        shutil.copy(source_path, dest_path)