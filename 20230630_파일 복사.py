import glob
import os
import shutil

path_src_nrrd = r'D:\00000000 Code\20230523_SwinUNETR_ICH\data_ICH\test\label_nrrd(until_5th)'
path_src_nifti = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
path_dst_nifti = r'D:\00000000 Code\20230523_SwinUNETR_ICH\data_ICH\test\image_Hemo_nifti_all'

nrrd_files = glob.glob(os.path.join(path_src_nrrd, '*.nrrd'))

for nrrd_file in nrrd_files:
    # 파일명에서 확장자를 제거
    base_name = os.path.basename(nrrd_file)
    file_name_without_ext = os.path.splitext(base_name)[0].replace('-label', '')

    # 동일한 이름의 '.nii.gz' 파일 찾기
    nii_file_path = os.path.join(path_src_nifti, file_name_without_ext + '.nii.gz')

    # 해당 파일이 B 폴더에 존재하는지 확인
    if os.path.exists(nii_file_path):
        # 동일한 파일명의 '.nii.gz' 파일을 C 폴더로 복사
        dest_path = os.path.join(path_dst_nifti, file_name_without_ext + '.nii.gz')
        shutil.copy2(nii_file_path, dest_path)
