# 2022.09.20
# - 확장자 없는 데이터의 확장자 변경

import pandas as pd
import shutil
import os
from glob import glob

path_root = r'Z:\Stroke\DATA\Demo_samples\태국\to_dcm\ST00001-AIDR\ST00001'
path_root = r'Z:\Stroke\DATA\Demo_samples\태국\to_dcm\ST00001-brain lesion\ST00001'
path_root = r'Z:\Stroke\DATA\Demo_samples\태국\to_dcm\ST00001-hemorrhage\ST00001'
path_root = r'Z:\Stroke\DATA\Demo_samples\태국\to_dcm\ST00001-NC\ST00001'

path_root = r'C:\Users\user\Downloads\Vitrea-brain lesion-20230106T024316Z-001\Vitrea-brain lesion\DICOM\ST00001'
path_root = r'Z:\Stroke\DATA\Demo_samples\Thailand\ST00001-AIDR\ST00001_dcm'

list_folder = os.listdir(path_root)

for cur_folder in list_folder:
    # 파일 리스트
    list_file = os.listdir(os.path.join(path_root, cur_folder))

    # 파일 별로 반복
    for cur_file in list_file:
        # 기존, 변경할 파일명/경로
        fn_before = cur_file
        fn_after = cur_file + '.dcm'

        # 파일명 변경
        os.rename(os.path.join(path_root, cur_folder, fn_before), os.path.join(path_root, cur_folder, fn_after))

#
# # 기관 별로 반복
# for cur_hospital in list_hospital:
#     # 현재 기관의 Data 폴더 루트
#     path_root = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\{}\Prep_DATA'.format(cur_hospital)
#     # 현재 기관의 환자 리스트
#     list_patient = os.listdir(path_root)
#
#     ##################################
#     ########## 폴더명 변경 #############
#     # 어쩌구 저쩌구 블라..... 아직 안함
#     ##################################
#
#     ##################################
#     ########## 파일명 변경 #############
#     # 환자 별로 파일명 변경
#     for cur_patient in list_patient:
#         ############################################################
#         # 1. 아주대
#         if cur_hospital == '1_AJUMC':
#             # A. LVO_UL_affined 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.dirname(cur_slice) + '\HR-A-0' + os.path.basename(cur_slice)
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#             # B. LVO_UL_affined_PADENET 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined_PADENET') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.dirname(cur_slice) + '\HR-A-0' + os.path.basename(cur_slice)
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#         ############################################################
#
#         ############################################################
#         # 2. 길병원
#         elif cur_hospital == '2_GMC':
#             # A. LVO_UL_affined 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.join(os.path.dirname(cur_slice), os.path.basename(cur_slice).replace('GH0', 'HR-G-0'))
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#             # B. LVO_UL_affined_PADENET 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined_PADENET') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.join(os.path.dirname(cur_slice), os.path.basename(cur_slice).replace('GH0', 'HR-G-0'))
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#         ############################################################
#
#         ############################################################
#         # 3. 이대병원
#         elif cur_hospital == '3_EUMC':
#             # A. LVO_UL_affined 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.join(os.path.dirname(cur_slice), os.path.basename(cur_slice).replace('E-', 'HR-E-0'))
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#             # B. LVO_UL_affined_PADENET 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined_PADENET') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.join(os.path.dirname(cur_slice), os.path.basename(cur_slice).replace('E-', 'HR-E-0'))
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#         ############################################################
#
#         ############################################################
#         # 4. CSUH
#         elif cur_hospital == '4_CSUH':
#             # A. LVO_UL_affined 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.join(os.path.dirname(cur_slice), os.path.basename(cur_slice).replace('C-', 'HR-C-0'))
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#             # B. LVO_UL_affined_PADENET 폴더
#             list_slice = glob(os.path.join(path_root, cur_patient, 'LVO_UL_affined_PADENET') + '/*.dcm')
#             for cur_slice in list_slice:
#                 dst = os.path.join(os.path.dirname(cur_slice), os.path.basename(cur_slice).replace('C-', 'HR-C-0'))
#                 print(dst)
#                 os.rename(cur_slice, dst)
#
#         ############################################################
#
#         ############################################################
#         # 6. CNUSH
#         elif cur_hospital == '6_SCHMC':
#             # 아무것도 안해도 됨, 이미 파일명 똑같음
#             pass
#
#         ############################################################
#
#         ############################################################
#         # 7. etc
#         elif cur_hospital == '7_etc':
#             # **********************************************************************
#             # 아예 복사해야됨.................. 이건 어찌 못함
#             # **********************************************************************
#             pass
#
#     ##################################