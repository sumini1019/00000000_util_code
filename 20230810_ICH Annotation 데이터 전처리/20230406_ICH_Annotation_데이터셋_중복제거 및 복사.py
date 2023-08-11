# 2023.04.06
# - 6차 Annotation 데이터셋에, 중복 데이터가 있음
# - 중복데이터 제외한 데이터만 리스트 뽑고나서,
# - 기존 데이터셋에 merge 하기 위한 목적

import os
import shutil

# A폴더와 B폴더의 경로
# A_dir = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230406_merged_dataset_until_5th"
# B_dir = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_6차 수령 데이터\ICH-A"
# dst_dir = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_6차 수령 데이터\ICH-A_exclude_duplicate"

A_dir = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_merged_dataset_until_6th"
B_dir = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230807_7차 수령 데이터\SDH-B"
dst_dir = B_dir + "_exclude_duplicate"

os.makedirs(dst_dir, exist_ok=True)

# 파일명의 앞 4자리와 확장자를 반환하는 함수
def get_partial_filename(filename):
    prefix = filename[:4]  # 첫 4자리
    ext = os.path.splitext(filename)[1]  # 확장자
    return prefix + ext

# A폴더와 B폴더의 파일 리스트 가져오기
A_files = os.listdir(A_dir)
B_files = os.listdir(B_dir)

# 파일명의 첫 4자리와 확장자를 기준으로 중복되는 파일 리스트 구하기
intersection_files = set([file for file in B_files if get_partial_filename(file) in [get_partial_filename(a_file) for a_file in A_files]])

# B폴더에만 있는 파일 리스트 구하기 (기준은 파일명의 첫 4자리와 확장자)
only_B_files = [file for file in B_files if get_partial_filename(file) not in [get_partial_filename(a_file) for a_file in A_files]]
only_B_files.sort()

only_B_files_err = []
for file in only_B_files:
    try:
        shutil.copy(os.path.join(B_dir, file), os.path.join(dst_dir, file))
    except shutil.SameFileError:
        print(f"[Error] '{file}' already exists in the destination folder")
        only_B_files_err.append(file)

print(f"\n[Summary] Copied {len(only_B_files) - len(only_B_files_err)} files to {dst_dir}")
print(f"[Summary] Failed to copy {len(only_B_files_err)} files: {only_B_files_err}")