# 2023.03.08
# - ICH Annotation이 폴더 별 정리되어 있음
    # - Subtype_Grade 형태
# - Subtype 기준 및 각 폴더의 파일 개수를 표로 생성하는 코드


import os
import csv
import matplotlib.pyplot as plt

# 폴더 경로 설정
path_root = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20220916_탑병원 의뢰 Annotation 데이터 및 프로그램 (Hemo, Normal 시리즈 기준)\2. 3D Annotation\1. Data_Series(Subtype, 등급별 분류)'


# 각 카테고리 별 파일 개수를 저장할 딕셔너리 생성
categories = {'EDH': 0, 'ICH': 0, 'IVH': 0, 'SAH': 0, 'SDH': 0}

# path_root 폴더 내의 모든 폴더에 대해 반복
for folder in os.listdir(path_root):
    # 각 폴더 내의 이미지 파일 개수를 세서 해당 카테고리에 더하기
    if folder.startswith('EDH'):
        categories['EDH'] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('ICH'):
        categories['ICH'] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('IVH'):
        categories['IVH'] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('SAH'):
        categories['SAH'] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('SDH'):
        categories['SDH'] += len(os.listdir(os.path.join(path_root, folder)))
    else:
        pass

# 결과를 csv 파일로 저장
with open('categories.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Category', 'File Count'])
    for category, count in categories.items():
        writer.writerow([category, count])

# 결과를 막대 그래프로 표시
plt.bar(categories.keys(), categories.values())
plt.title('File Count by Category')
plt.xlabel('Category')
plt.ylabel('File Count')
plt.show()


# 카테고리 별 등급별 파일 개수를 저장할 딕셔너리 생성
categories = {'EDH': {}, 'ICH': {}, 'IVH': {}, 'SAH': {}, 'SDH': {}}

# path_root 폴더 내의 모든 폴더에 대해 반복
for folder in os.listdir(path_root):
    # 각 폴더 내의 이미지 파일 개수를 세서 해당 카테고리와 등급에 더하기
    if folder.startswith('EDH'):
        grade = folder.split('_')[1]
        if grade not in categories['EDH']:
            categories['EDH'][grade] = 0
        categories['EDH'][grade] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('ICH'):
        grade = folder.split('_')[1]
        if grade not in categories['ICH']:
            categories['ICH'][grade] = 0
        categories['ICH'][grade] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('IVH'):
        grade = folder.split('_')[1]
        if grade not in categories['IVH']:
            categories['IVH'][grade] = 0
        categories['IVH'][grade] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('SAH'):
        grade = folder.split('_')[1]
        if grade not in categories['SAH']:
            categories['SAH'][grade] = 0
        categories['SAH'][grade] += len(os.listdir(os.path.join(path_root, folder)))
    elif folder.startswith('SDH'):
        grade = folder.split('_')[1]
        if grade not in categories['SDH']:
            categories['SDH'][grade] = 0
        categories['SDH'][grade] += len(os.listdir(os.path.join(path_root, folder)))
    else:
        pass

# 각 카테고리별 subplot을 생성하고 각 subplot에 막대 그래프 추가
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8, 20))

for i, category in enumerate(categories.keys()):
    axs[i].bar(categories[category].keys(), categories[category].values())
    axs[i].set_title(f'File Count by Grade - {category}')
    axs[i].set_xlabel('Grade')
    axs[i].set_ylabel('File Count')

# 간격 조절
plt.subplots_adjust(hspace=0.6)

# 그래프 출력
plt.show()


# 각 폴더 별 파일 개수를 저장할 딕셔너리 생성
folder_counts = {}

# path_root 폴더 내의 모든 폴더에 대해 반복
for folder in os.listdir(path_root):
    # 각 폴더 내의 이미지 파일 개수를 세서 딕셔너리에 추가
    folder_path = os.path.join(path_root, folder)
    if os.path.isdir(folder_path):
        file_count = len(os.listdir(folder_path))
        folder_counts[folder] = file_count

# 결과를 csv 파일로 저장
with open('folder_counts.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Folder', 'File Count'])
    for folder, count in folder_counts.items():
        writer.writerow([folder, count])

# 결과를 막대 그래프로 표시
plt.bar(folder_counts.keys(), folder_counts.values())
plt.title('File Count by Folder')
plt.xlabel('Folder')
plt.ylabel('File Count')
plt.xticks(rotation=90)
plt.show()



# 4. 등급 (_A, _B, ..., _G) 별 파일 개수 (EDH / ICH/ ... / SDH 관계 없이 통합)
import pandas as pd
folders = os.listdir(path_root)
counts_by_grade_all_types = {}

for folder in folders:
    if os.path.isdir(os.path.join(path_root, folder)):
        folder_grade = folder.split("_")[1]
        if folder_grade in counts_by_grade_all_types:
            counts_by_grade_all_types[folder_grade] += len(os.listdir(os.path.join(path_root, folder)))
        else:
            counts_by_grade_all_types[folder_grade] = len(os.listdir(os.path.join(path_root, folder)))

# 결과를 csv 파일로 저장
df4 = pd.DataFrame.from_dict(counts_by_grade_all_types, orient="index", columns=["Count"])
df4.index.name = "Grade"
df4.to_csv("counts_by_grade_all_types.csv")

# 결과를 plot으로 시각화
plt.bar(df4.index, df4["Count"])
plt.title("File Counts by Grade (All Types)")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.savefig("counts_by_grade_all_types.png")
plt.show()


# # 각 폴더에 대한 파일 개수를 저장할 딕셔너리 초기화
# folder_file_counts = {
#     "EDH": 0, "ICH": 0, "IVH": 0, "SAH": 0, "SDH": 0
# }
#
# # 모든 폴더에 대한 파일 개수를 저장할 딕셔너리 초기화
# all_folder_file_counts = {}
#
#
# # 폴더 내부의 파일 개수를 계산하는 함수
# def count_files(folder_path):
#     count = 0
#     for filename in os.listdir(folder_path):
#         if os.path.isfile(os.path.join(folder_path, filename)):
#             count += 1
#     return count
#
#
# # 폴더 내부의 파일 개수를 계산하고 딕셔너리에 저장
# for foldername in os.listdir(path_root):
#     if os.path.isdir(os.path.join(path_root, foldername)):
#         folder_path = os.path.join(path_root, foldername)
#         count = count_files(folder_path)
#         folder_name_split = foldername.split("_")
#         grade = folder_name_split[-1]
#         type_name = "_".join(folder_name_split[:-1])
#
#         # 폴더 이름을 이용해 각 등급에 대한 파일 개수를 저장
#         if type_name in folder_file_counts:
#             folder_file_counts[type_name] += count
#
#         # 모든 폴더에 대한 파일 개수를 저장
#         all_folder_file_counts[foldername] = count
#
# # 첫번째 표: 등급 별 파일 개수를 저장한 딕셔너리를 csv 파일로 저장
# with open("table1.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Type", "File Count"])
#     for key, value in folder_file_counts.items():
#         writer.writerow([key, value])
#
# # 두번째 표: 모든 폴더 별 파일 개수를 저장한 딕셔너리를 csv 파일로 저장
# with open("table2.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Folder Name", "File Count"])
#     for key, value in all_folder_file_counts.items():
#         writer.writerow([key, value])
