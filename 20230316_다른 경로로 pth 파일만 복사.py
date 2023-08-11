# import os
# import shutil
#
# path_src = r'D:\00000000_Model\20230217_Heuron_Stroke_v012_5fold_서버학습모델\0_HeuronStroke_v012_weight'
# path_dst = r'D:\OneDrive\00000000_Code\20221102_cSTROKE\0_engine_version\cSTROKE\git_new\Heuron_ELVO\weights'
#
# # path_dst 디렉토리가 없으면 생성
# if not os.path.exists(path_dst):
#     os.makedirs(path_dst)
#
# for root, dirs, files in os.walk(path_src):
#     for file in files:
#         if file.endswith('.pth'):
#             src_file = os.path.join(root, file)
#             dst_file = src_file.replace(path_src, path_dst)
#             dst_folder = os.path.dirname(dst_file)
#
#             # dst_folder 디렉토리가 없으면 생성
#             if not os.path.exists(dst_folder):
#                 os.makedirs(dst_folder)
#
#             # 파일 복사
#             shutil.copy2(src_file, dst_file)



import os

def rename_files(path_dir, target_substring):
    # 해당 경로의 모든 파일명을 불러옴
    filenames = os.listdir(path_dir)

    for filename in filenames:
        # 파일명에 target_substring이 포함되어 있다면, target_substring으로 시작하는 부분을 파일명에서 제거
        if target_substring in filename:
            new_filename = filename[:filename.index(target_substring)]
            # 파일명에서 target_substring 이전의 부분만 남기고 확장자를 포함한 나머지 부분을 추가
            ext = os.path.splitext(filename)[1]
            new_filename += ext
            # 파일명 변경
            os.rename(os.path.join(path_dir, filename), os.path.join(path_dir, new_filename))




path_src = r'D:\OneDrive\00000000_Code\20221102_cSTROKE\0_engine_version\cSTROKE\git_new\Heuron_ELVO\weights'

list_dir = os.listdir(path_src)

for cur_dir in list_dir:
    rename_files(os.path.join(path_src, cur_dir), '_iter')