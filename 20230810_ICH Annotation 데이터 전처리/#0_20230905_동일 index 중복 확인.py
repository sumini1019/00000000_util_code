# 2023.09.05
# - 수령 데이터셋에, 동일 index이나 중복해서 Annotation 한 데이터가 존재함
# - 해당 데이터들 확인하고, 수동으로 제거하기 위해 리스트를 뽑는 과정
#
# **********************************************************************
# **************  리스트를 확인하고, 수동으로 확인 후 제거 *******************
# **********************************************************************

import os
from collections import defaultdict

folder_path = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230901_8차 수령 데이터\SAH-B'

def find_duplicate_index_files(folder_path):
    index_files = defaultdict(list)

    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            filename, extension = os.path.splitext(file)
            if len(filename) >= 4:
                index = filename[:4]
                index_files[(index, extension)].append(file)

    duplicate_files = [(index, extension, files) for (index, extension), files in index_files.items() if len(files) > 1]

    return duplicate_files

duplicate_files = find_duplicate_index_files(folder_path)

if duplicate_files:
    print("동일한 인덱스와 확장자를 가진 파일들:")
    for index, extension, files in duplicate_files:
        print(f"인덱스: {index}, 확장자: {extension}")
        for file in files:
            print(file)
        print("-" * 30)
else:
    print("동일한 인덱스와 확장자를 가진 파일이 없습니다.")