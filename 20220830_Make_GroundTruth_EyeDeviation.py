# 2022.08.30
# - Eye Deviation 결과 csv 파일로부터
# - 환자 별, Eye deviation Ground Truth CSV 파일을 하나 생성하는 것


import numpy as np
import shutil
import os
import pandas as pd
from glob import glob

# 환자 별 ED csv 파일 경로
path_ED_csv = 'Z:\Stroke\cELVO\GT_GUI_2022\Eye_Dev\Eval_Results'
# path_ED_csv = 'Z:\Stroke\cELVO\GT_GUI_2022\Eye_Dev\Eval_Results_GMC'

# Ground Truth CSV 저장할 경로
path_result_csv = r''

# csv 리스트
# list_csv = os.listdir(path_ED_csv)
list_csv = glob(path_ED_csv + '/*.csv')


# Eye Deviation 클래스 분류
def classify_ED(deg):
    # EyeBall / Lens 못 찾은 경우에 대한 예외처리
    if deg == -1000:
        result_ED = 'Undetected'
    else:
        # ED 존재 range
        # -> 15~60   or   20~60
        if (abs(deg) > 15) or (abs(deg) > 60):
            # 음수
            if deg < 0:
                result_ED = 'ED_Left'
            # 양수
            else:
                result_ED = 'ED_Right'
        # Normal (ED X)
        else:
            result_ED = 'Normal'

    return result_ED

# Left / Right 별, Max Degree 계산 함수
def get_max_degree(df):
    # Left / Right 데이터프레임 중, Direction이 Left/Right 인 것만 골라오기
    df_Left = df[(df['Left Lens - Direction'] == 'Left')
                 | (df['Left Lens - Direction'] == 'Right')
                 | (df['Left Lens - Direction'] == 'Front')].reset_index(
        drop=True)
    df_Right = df[(df['Right Lens - Direction'] == 'Left')
                  | (df['Right Lens - Direction'] == 'Right')
                  | (df['Right Lens - Direction'] == 'Front')].reset_index(
        drop=True)

    ### Left
    # Direction에 해당하는 행이 0개일 경우, 못 찾은 것이므로 예외처리 (-1000)
    if len(df_Left) == 0:
        max_deg_Left = -1000
    else:
        ##########################################
        # Max Degree 찾기
        # 1. 절대값이 가장 큰 Degree 찾기
        # 2. Left일 경우, 음수값으로 변경

        # 절대값이 가장 큰 degree의 index
        max_idx_Left = df_Left['Left Lens - Angle'].astype('float').idxmax()
        # 해당 index의 방향 확인
        max_direction_Left = df_Left['Left Lens - Direction'][max_idx_Left]
        # 방향에 따라, max degree 계산
        if max_direction_Left == 'Left':
            max_deg_Left = df_Left['Left Lens - Angle'].astype('float')[max_idx_Left] * -1
        elif max_direction_Left == 'Right':
            max_deg_Left = df_Left['Left Lens - Angle'].astype('float')[max_idx_Left]
        elif max_direction_Left == 'Front':
            max_deg_Left = df_Left['Left Lens - Angle'].astype('float')[max_idx_Left]
        else:
            print('Error : 디렉션이 잘못 됨')


    ### Right
    # Direction에 해당하는 행이 0개일 경우, 못 찾은 것이므로 예외처리 (-1000)
    if len(df_Right) == 0:
        max_deg_Right = -1000
    else:
        ##########################################
        # Max Degree 찾기
        # 1. 절대값이 가장 큰 Degree 찾기
        # 2. Left일 경우, 음수값으로 변경

        # 절대값이 가장 큰 degree의 index
        max_idx_Right = df_Right['Right Lens - Angle'].astype('float').idxmax()
        # 해당 index의 방향 확인
        max_direction_Right = df_Right['Right Lens - Direction'][max_idx_Right]
        # 방향에 따라, max degree 계산
        if max_direction_Right == 'Left':
            max_deg_Right = df_Right['Right Lens - Angle'].astype('float')[max_idx_Right] * -1
        elif max_direction_Right == 'Right':
            max_deg_Right = df_Right['Right Lens - Angle'].astype('float')[max_idx_Right]
        elif max_direction_Right == 'Front':
            max_deg_Right = df_Right['Right Lens - Angle'].astype('float')[max_idx_Right]
        else:
            print('Error : 디렉션이 잘못 됨')

    # 리턴
    return max_deg_Left, max_deg_Right

# 방향 고려 못한 잘못된 함수 (폐기)
def get_rep_degree(df):
    # Left/Right 별, 평균 / Max Degree 계산
    # 1. Left
    list_deg_LE = list((df['Left Lens - Angle']).dropna())
    while '-' in list_deg_LE:
        list_deg_LE.remove('-')
    # deg 값이 하나도 없는 경우 확인
    if len(list_deg_LE) > 0:
        list_deg_LE = list(map(float, list_deg_LE))
        avg_deg_LE = np.mean(np.array(list_deg_LE))
        max_deg_LE = np.max(np.array(list_deg_LE))
    else:
        # deg 값이 없다는건, 측정이 불가한 것이므로 -1 로 예외처리
        avg_deg_LE = -1
        max_deg_LE = -1
    # 2. Right
    list_deg_RE = list((df['Right Lens - Angle']).dropna())
    while '-' in list_deg_RE:
        list_deg_RE.remove('-')
    # deg 값이 하나도 없는 경우 확인
    if len(list_deg_RE) > 0:
        list_deg_RE = list(map(float, list_deg_RE))
        avg_deg_RE = np.mean(np.array(list_deg_RE))
        max_deg_RE = np.max(np.array(list_deg_RE))
    else:
        # deg 값이 없다는건, 측정이 불가한 것이므로 -1 로 예외처리
        avg_deg_RE = -1
        max_deg_RE = -1


# 결과 DataFrame
df_result = pd.DataFrame(columns=['ID', 'LE_ED_Cls', 'RE_ED_Cls', 'LE_ED_MaxDeg', 'RE_ED_MaxDeg'])

# 모든 CSV에 대해 순회
for cur_csv in list_csv:
    # csv 로드
    cur_df = pd.read_csv(cur_csv)

    # Left/Right 별, 평균 / Max Degree 계산 (ver2)
    max_deg_LE, max_deg_RE = get_max_degree(cur_df)

    # 평균 / Max Degree 값에 따라, 클래스 분류 (ED_Left, ED_Right, Normal)
    cls_LE = classify_ED(max_deg_LE)
    cls_RE = classify_ED(max_deg_RE)

    # 결과 DataFrame에 누적
    cur_result = {'ID': os.path.basename(cur_csv)[:6],
                  'LE_ED_Cls': cls_LE, 'RE_ED_Cls': cls_RE,
                  'LE_ED_MaxDeg': max_deg_LE, 'RE_ED_MaxDeg': max_deg_RE}
    df_result = df_result.append(cur_result, ignore_index=True)

# 결과 DataFrame 저장
print(df_result)
df_result.to_csv('GT_EyeDeviation_20221021.csv', index=False)