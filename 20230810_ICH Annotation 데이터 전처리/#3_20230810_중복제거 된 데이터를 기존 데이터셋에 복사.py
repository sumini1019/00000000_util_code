# 2023.08.10
# - 7차 수령 데이터에서 중복 제거한 데이터의 각 폴더를,
# - 기존 1~6차 Merge 한 데이터셋에 복사
# - 중복이 있는 경우 알림

import os
import shutil

# 소스 폴더 경로들
src_paths = [
    r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_8차 수령 데이터\SAH-A_exclude_duplicate",
    r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_8차 수령 데이터\SAH-B_exclude_duplicate",
]

# 대상 폴더 경로
dst_path_before = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230810_merged_dataset_until_7th'
dst_path = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_merged_dataset_until_8th\new"

# 기존 데이터셋을 복사
# shutil.copytree(dst_path_before, dst_path)

# 대상 폴더에 이미 있는 파일들의 인덱스 목록 생성
existing_indices = {filename[:4] for filename in os.listdir(dst_path)}

for src_path in src_paths:
    for file in os.listdir(src_path):
        file_index = file[:4]  # 파일의 앞 4자리로 인덱스 추출

        # 동일한 인덱스의 파일이 대상 폴더에 이미 있는 경우 경고 메시지 출력
        if file_index in existing_indices:
            print(f"Warning! File with index {file_index} already exists in the destination folder.")

        # 파일 복사
        shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, file))