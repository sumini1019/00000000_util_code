### 목적 ###
# - 타겟 폴더 내에서, patient 마다
# - CTP 폴더가 존재하는지 확인하고
# - CBV, CBF 폴더가 없는지 확인

# - 그 후, CTP 폴더 내에 이동시키고

# - 폴더 리스트대로, 모든 폴더를 만들기

import shutil
import os
import pandas as pd


# 타겟 경로
path_target = r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\2_GMC\F_DATA\ID_Renewal'

# 환자 폴더 리스트
list_dir_patient = os.listdir(path_target)

# 환자 폴더 선회
for cur_dir in list_dir_patient:
    # 해당 Patient의 폴더 리스트 (영상 Type별)
    list_dir_type = os.listdir(os.path.join(path_target, cur_dir))


    # # 1.
    # ##################################################
    # # 타입 리스트에 해당하는 폴더가 있는지 확인
    # list_type = ['CBV', 'CBF', 'FED', 'MTT', 'PENUMBRA', 'TMAX', 'TTD']
    #
    # for cur_type in list_dir_type:
    #     if cur_type in list_type:
    #         print('환자 : {}'.format(cur_dir))
    #         print('Type : {}\n'.format(cur_type))
    # ####################################################
    # # -> 결과 : CBV, CBF  가 꽤 많음
    # # -> 수정 후 : 이제 없음


    # # 2.
    # ##################################################
    # # CTP 폴더도 있나?
    # # 타입 리스트에 해당하는 폴더가 있는지 확인
    # list_type = ['CTP']
    #
    # for cur_type in list_dir_type:
    #     if cur_type in list_type:
    #         print('환자 : {}'.format(cur_dir))
    #         print('Type : {}\n'.format(cur_type))
    # ##################################################
    # # -> 결과 : CTP 폴더도 많음



    # # 3.
    # # CTP 있으면서, CBF나, CBV가 같이 있는 경우도 있나?
    # list_type = ['CBV', 'CBF', 'FED', 'MTT', 'PENUMBRA', 'TMAX', 'TTD']
    #
    # # CTP 있고,
    # if 'CTP' in list_dir_type:
    #     # CTP의 서브타입들이 안에 있나?
    #     for cur_type in list_type:
    #         if cur_type in list_dir_type:
    #             print('환자 : {}'.format(cur_dir))
    #             print('Type : {}\n'.format(cur_type))
    #
    # # 결과 : 많음;;;;;;;;
    # # 수정 후 : 이제 없음


    # 4. 결론
    # - list_type에 존재하는게 있으면
    list_type = ['CBV', 'CBF', 'FED', 'MTT', 'PENUMBRA', 'TMAX', 'TTD']
    for cur_type in list_dir_type:
        if cur_type in list_type:
            #   - CTP 폴더가 있으면, 기존 폴더에 복사
            if 'CTP' in list_dir_type:
                # - 기존 폴더에 type 폴더가 있으면?
                if os.path.isdir(os.path.join(path_target, cur_dir, 'CTP', cur_type)):
                    # 폴더 내에 자료가 없으면 이동
                    if len(os.listdir(os.path.join(path_target, cur_dir, 'CTP', cur_type))) == 0:
                        # 1. 경로
                        path_subtype_exist = os.path.join(path_target, cur_dir, cur_type)  # 원본 경로
                        path_subtype_dest = os.path.join(path_target, cur_dir, 'CTP', cur_type)  # 목적 경로
                        # 2. 이동
                        if os.path.isdir(path_subtype_dest):
                            shutil.rmtree(path_subtype_dest)
                        shutil.move(path_subtype_exist, path_subtype_dest)

                    # 폴더 내에 자료가 있으면, 원래 타입의 데이터 삭제
                    else:
                        shutil.rmtree(os.path.join(path_target, cur_dir, cur_type))  # 원본 경로
                # - 기존 폴더에 type 폴더가 없으면, 그대로 이동
                else:
                    # 1. 경로
                    path_subtype_exist = os.path.join(path_target, cur_dir, cur_type)  # 원본 경로
                    path_subtype_dest = os.path.join(path_target, cur_dir, 'CTP', cur_type)  # 목적 경로
                    # 2. 이동
                    if os.path.isdir(path_subtype_dest):
                        shutil.rmtree(path_subtype_dest)
                    shutil.move(path_subtype_exist, path_subtype_dest)

                # 모든 CTP Sub 폴더 생성
                for sub_type in list_type:
                    os.makedirs(os.path.join(os.path.join(path_target, cur_dir, 'CTP'), sub_type), exist_ok=True)

            # - CTP 폴더가 없으면, 새로 생성 후 복사
            else:
                # CTP 폴더 생성
                path_CTP = os.path.join(path_target, cur_dir, 'CTP')
                os.makedirs(path_CTP, exist_ok=True)

                # 존재하던 sub_type 폴더 이동
                # 1. 경로
                path_subtype_exist = os.path.join(path_target, cur_dir, cur_type)       # 원본 경로
                path_subtype_dest = os.path.join(path_target, cur_dir, 'CTP', cur_type) # 목적 경로
                # 2. 이동
                if os.path.isdir(path_subtype_dest):
                    shutil.rmtree(path_subtype_dest)
                shutil.move(path_subtype_exist, path_subtype_dest)

                # 모든 CTP Sub 폴더 생성
                for sub_type in list_type:
                    os.makedirs(os.path.join(path_CTP, sub_type), exist_ok=True)



