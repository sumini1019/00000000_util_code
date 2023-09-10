# _1-label.nrrd, -2-label.nrrd  와 같이 끝나는 데이터의 이름을 변경

import os
import shutil

# 여러 개의 타겟 폴더 설정
src_paths = [
    r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_8차 수령 데이터\SAH-A_exclude_duplicate",
    r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_8차 수령 데이터\SAH-B_exclude_duplicate",
]

def rename_files_in_directory(B_dir):
    B_files = os.listdir(B_dir)

    renamed_files = []

    # 파일명에서 확장자 제거 및 검사
    for file in B_files:
        if file.endswith('.nii.gz'):
            file_without_ext = file[:-7]  # 확장자 .nii.gz 제거
        elif file.endswith('.nrrd'):
            file_without_ext = file[:-5]  # 확장자 .nrrd 제거
        else:
            continue

        # `_1-label`, `_2-label`, `_3-label`, `_4-label`로 끝나는 파일명 확인 및 수정
        if file_without_ext.endswith(('_1-label', '_2-label', '_3-label', '_4-label')):
            new_name = file_without_ext.replace('_1-label', '-label').replace('_2-label', '-label').replace('_3-label', '-label').replace('_4-label', '-label')
            if file.endswith('.nii.gz'):
                new_name += '.nii.gz'
            else:
                new_name += '.nrrd'

            shutil.move(os.path.join(B_dir, file), os.path.join(B_dir, new_name))
            renamed_files.append((file, new_name))

    print(f"Folder: {B_dir}")
    print("Renamed Files:")
    for old, new in renamed_files:
        print(f"{old} -> {new}")

    print(f"\nTotal renamed files in this folder: {len(renamed_files)}")
    print("-------------------------------------------------------------")


for path in src_paths:
    rename_files_in_directory(path)
