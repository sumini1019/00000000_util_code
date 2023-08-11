import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
import random
import shutil
import os
from glob import glob
import pydicom
import copy

# Train / test 모드
mode = 'train' #'test'

# 데이터 타입
# type_data = 'hemi_original'
type_data = 'hemi_original_PADENET'

if mode == 'train':
    # 전체 root 경로
    path_root = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET'
    # DMS / Normal 나눠서 저장할 경로
    path_save_split = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TrainSET'
    # Slice 별 Label 정보
    df_label = pd.read_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET_Annot_v211022.csv')
else:
    # 전체 root 경로
    path_root = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TestSET'
    # DMS / Normal 나눠서 저장할 경로
    path_save_split = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TestSET'
    # Slice 별 Label 정보
    df_label = pd.read_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TestSET_Annot_v211022.csv')

# 변경된 파일명 및 index로 저장할 Dataframe 생성
df_hemi = pd.DataFrame(columns=df_label.columns)
df_wb = pd.DataFrame(columns=df_label.columns)

# Series 폴더 리스트
list_series = sorted(os.listdir(path_root))[732:]

# 에러 카운트 및 리스트
list_error_cnt = []

# 데이터프레임 누적 카운트
cnt_df = 0

# 각 폴더마다, 순회
for cur_series in list_series:
    # 카운트
    error_cnt = 0

    # 폴더의 Slice 리스트 읽어오기
    list_slice = sorted(os.listdir(os.path.join(path_root, cur_series, type_data)))
    list_slice = [file for file in list_slice if file.endswith(".dcm")]

    # dms 있는 Slice 리스트만 파싱
    list_slice_dms = [file for file in list_slice if 'dms' in file]

    # index 리스트 (중복 제거)
    list_index = [file[:3] for file in list_slice]
    list_index = sorted(list(set(list_index)))

    # 동일한 index의 L / R 이미지 읽어오기
    for cur_index in list_index:
        # 해당 index의 파일 리스트
        index_file = [file for file in list_slice if cur_index in file]
        # Left / Right 이미지 읽기
        if len(index_file) == 2:
            # 현재 파일명
            if '_L' in index_file[0]:
                name_left = index_file[0]
                name_right = index_file[1]
            else:
                name_left = index_file[1]
                name_right = index_file[0]

            # 저장용 새로운 파일명
            name_left_for_save = cur_series + '_' + name_left
            name_right_for_save = cur_series + '_' + name_right
        else:
            print("Error - Series{} index{} 해당하는 파일이 2개가 아님".format(cur_series, cur_index))
            error_cnt = error_cnt + 1
            continue

        # dcm 경로
        path_dcm_left = os.path.join(path_root, cur_series, type_data, name_left)
        path_dcm_right = os.path.join(path_root, cur_series, type_data, name_right)

        # dcm 로드
        dcm_left = pydicom.read_file(path_dcm_left)
        dcm_right = pydicom.read_file(path_dcm_right)

        # Left / Right Array
        array_left = dcm_left.pixel_array
        array_right = dcm_right.pixel_array

        # # Windowing
        # array_left = np.where(array_left < 0, 0, array_left)
        # array_right = np.where(array_right < 0, 0, array_right)

        # Array 합성
        array_sum = array_left + array_right
        # dcm 파일 생성
        dcm_sum = copy.deepcopy(dcm_left)
        # Array 업데이트
        dcm_sum.PixelData = array_sum.tobytes()

        # 저장 경로
        if mode == 'train':
            path_save = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET\{}\{}_WB'.format(cur_series, type_data)
        else:
            path_save = r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TestSET\{}\{}_WB'.format(cur_series, type_data)

        if not os.path.isdir(path_save):
            os.mkdir(path_save)

        # sum 완료된 dcm 저장
        name_WB = name_left.replace("_L.", "_WB.")
        name_WB_for_save = name_left_for_save.replace("_L.", "_WB.")
        path_save_for_sum = os.path.join(path_save, name_WB)
        dcm_sum.save_as(path_save_for_sum)


        ###################################################################################
        # file name 기반으로, DMS 여부 확인 후 개별 폴더에 저장 (dms / normal)
        # 현재 ID에 해당하는 Dataframe
        cur_df = df_label[df_label['ID'] == cur_series]
        # 현재 index에 해당하는 Dataframe
        index_df = cur_df[cur_df['ID_Filename'].cur_folder.contains(cur_index + '.dcm')].reset_index(drop=True)
        # Left / Right 반구의 DMS Label
        label_LH = index_df['DMS Eval. - LH'][0]
        label_RH = index_df['DMS Eval. - RH'][0]


        # DMS 여부에 따른 파일 복사 (반구 데이터) - PADE-Net 데이터 여부에 따라 분기
        if type_data == 'hemi_original':
            if label_LH == 1:
                shutil.copyfile(path_dcm_left, os.path.join(path_save_split, 'Hemi', 'dms', name_left_for_save))
            else:
                shutil.copyfile(path_dcm_left, os.path.join(path_save_split, 'Hemi', 'normal', name_left_for_save))
            if label_RH == 1:
                shutil.copyfile(path_dcm_right, os.path.join(path_save_split, 'Hemi', 'dms', name_right_for_save))
            else:
                shutil.copyfile(path_dcm_right, os.path.join(path_save_split, 'Hemi', 'normal', name_right_for_save))
            # DMS 여부에 따른 파일 복사 (Whole Brain)
            if label_LH or label_RH:
                shutil.copyfile(path_save_for_sum, os.path.join(path_save_split, 'WholeBrain', 'dms', name_WB_for_save))
            else:
                shutil.copyfile(path_save_for_sum, os.path.join(path_save_split, 'WholeBrain', 'normal', name_WB_for_save))

        else:
            if label_LH == 1:
                shutil.copyfile(path_dcm_left, os.path.join(path_save_split, 'Hemi_PADE', 'dms', name_left_for_save))
            else:
                shutil.copyfile(path_dcm_left, os.path.join(path_save_split, 'Hemi_PADE', 'normal', name_left_for_save))
            if label_RH == 1:
                shutil.copyfile(path_dcm_right, os.path.join(path_save_split, 'Hemi_PADE', 'dms', name_right_for_save))
            else:
                shutil.copyfile(path_dcm_right, os.path.join(path_save_split, 'Hemi_PADE', 'normal', name_right_for_save))
            # DMS 여부에 따른 파일 복사 (Whole Brain)
            if label_LH or label_RH:
                shutil.copyfile(path_save_for_sum, os.path.join(path_save_split, 'WholeBrain_PADE', 'dms', name_WB_for_save))
            else:
                shutil.copyfile(path_save_for_sum, os.path.join(path_save_split, 'WholeBrain_PADE', 'normal', name_WB_for_save))


# for cur_series in list_series:
#     for cur_index in list_index:
        ############### 바뀐 파일명에 맞게, Dataframe 새로 생성 #####################

        # 2. label 정보에 따라, 각 column 행을 추가
        # 2.1 - Hemisphere (Left)
        index_df_new = index_df.copy()
        index_df_new.loc[0, 'ID_Filename'] = name_left_for_save
        df_hemi = pd.concat([df_hemi, index_df_new])
        # 2.2 - Hemisphere (Right)
        index_df_new.loc[0, 'ID_Filename'] = name_right_for_save
        df_hemi = pd.concat([df_hemi, index_df_new])
        # 2.3 - Whole Brain
        index_df_new.loc[0, 'ID_Filename'] = name_WB_for_save
        df_wb = pd.concat([df_wb, index_df_new])

        # 3. 데이터프레임 저장
        if mode == 'train':
            df_hemi.to_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TrainSET\df_hemi.csv', index=False)
            df_wb.to_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TrainSET\df_wb.csv', index=False)
        else:
            df_hemi.to_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TestSET\df_hemi.csv', index=False)
            df_wb.to_csv(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\DMS 여부 기준 데이터셋\TestSET\df_wb.csv', index=False)

        # # index_df 를 계속 누적하되, ID_Filename 값만 변경
        #
        # # index_df 의 ID_Filename 값 변경
        # index_df_new = index_df.copy()
        # index_df_new.loc[0, 'ID_Filename'] = name_WB.replace("_WB.", ".")
        #
        # # 카운트에 따라, copy or 병합
        # if cnt_df == 0:
        #     df_new = index_df_new.copy()
        # else:
        #     df_new = pd.concat([df_new, index_df_new])
        #
        # # 데이터프레임 누적 카운트 증가
        # cnt_df = cnt_df + 1

    # error cnt 누적
    list_error_cnt.append(error_cnt)

# 에러 출력
for i in range(0, len(list_series)):
    print('Patient {} : Error Cnt {}'.format(list_series[i], list_error_cnt[i]))