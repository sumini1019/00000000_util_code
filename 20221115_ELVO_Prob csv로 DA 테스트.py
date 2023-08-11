import os
import pydicom
from glob import glob
import pandas as pd
import shutil

import sys

from logger.LogModule import LogModule

# Logger
logger = LogModule().getLogger()
logger_with_file = LogModule().getFileLogger()


def DecisionAlgo_ELVO(result_LVO=None, result_ED=None, inst_GIL=False):
    # Threshold (길병원 Triage 임상용)
    if inst_GIL:
        LVO_1st = 0.98
        LVO_2nd_Upper = 0.5
        LVO_2nd_Lower = 0.1
        LVO_2nd_DMS = 0.8
        LVO_3rd = 0.95
        EIC_3rd = 0.95
        LVO_4th = 0.85
        DMS_4th = 0.95

    # Threshold (Default)
    else:
        LVO_1st = 0.98
        LVO_2nd_Upper = 0.5
        LVO_2nd_Lower = 0.1
        LVO_3rd = 0.6
        EIC_3rd = 0.9
        LVO_4th = 0.7
        DMS_4th = 0.8

    ############################
    # 1. Eye Deviation 결과 산출
    ############################
    def Diagnosis_ED(result_LVO, result_ED):
        ED_Cls_LH = result_ED['ED_Cls_Left']
        ED_Cls_RH = result_ED['ED_Cls_Right']

        ### 1. ED Class에 따른, 최종 ED 결과
        if (ED_Cls_LH == 'ED_R') and (ED_Cls_RH == 'ED_R'):
            Diag_ED = 'RB'
        elif (ED_Cls_LH == 'ED_L') and (ED_Cls_RH == 'ED_L'):
            Diag_ED = 'LB'
        elif (ED_Cls_LH == 'ED_R') and (ED_Cls_RH == 'ED_L'):
            Diag_ED = 'NG'
        elif (ED_Cls_LH == 'ED_L') and (ED_Cls_RH == 'ED_R'):
            Diag_ED = 'NG'
        elif (ED_Cls_LH == 'ED_R') or (ED_Cls_RH == 'ED_R'):
            Diag_ED = 'RU'
        elif (ED_Cls_LH == 'ED_L') or (ED_Cls_RH == 'ED_L'):
            Diag_ED = 'LU'
        elif (ED_Cls_LH == 'Normal') and (ED_Cls_RH == 'Normal'):
            Diag_ED = 'NG'
        else:
            logger_with_file.debug('Error - cELVO Decision Algorithm')

        ### 2. Eye Deviation 결과 보완
        # - ED가 한쪽 안구만 왔을 때, (RU / LU)
        # - LVO Prob 확인해서, ED 결과 (Diag_ED)를 수정
        #  -> 반대 반구의 LVO Prob이 높으면, ED가 틀렸다고 생각하고 수정하기 위함
        if (Diag_ED == 'RU') or (Diag_ED == 'LU'):
            if (Diag_ED == 'LU') and (result_LVO['Prob_LVO_LH'] < 0.1) and (result_LVO['Prob_LVO_RH'] >= 0.95):
                Diag_ED = 'RU'
            elif (Diag_ED == 'RU') and (result_LVO['Prob_LVO_LH'] >= 0.95) and (result_LVO['Prob_LVO_RH'] < 0.1):
                Diag_ED = 'LU'

        return Diag_ED

    Diag_ED = Diagnosis_ED(result_LVO, result_ED)

    ############################
    # 2. LVO 결과 Decision
    #   -> ED 결과에 따른 분기
    ############################
    def Diagnosis_LVO(result_LVO, Diag_ED):
        ### A. ED 결과 Normal (NG)인 경우,
        #   -> 양 반구에 대해, 각각 LVO 여부 검사
        if Diag_ED == 'NG':
            ### 1. 좌반구 분류
            # - 조건 2개 중 하나라도 만족하면 양성
            if (result_LVO['Prob_LVO_LH'] >= LVO_3rd) and (result_LVO['Prob_EIC_LH'] >= EIC_3rd):
                Diag_LVO_LH = 1
            elif (result_LVO['Prob_LVO_LH'] >= LVO_4th) and (result_LVO['Prob_DMS_LH'] >= DMS_4th):
                Diag_LVO_LH = 1
            else:
                Diag_LVO_LH = 0

            ### 2. 우반구 분류
            # - 조건 2개 중 하나라도 만족하면 양성
            if (result_LVO['Prob_LVO_RH'] >= LVO_3rd) and (result_LVO['Prob_EIC_RH'] >= EIC_3rd):
                Diag_LVO_RH = 1
            elif (result_LVO['Prob_LVO_RH'] >= LVO_4th) and (result_LVO['Prob_DMS_RH'] >= DMS_4th):
                Diag_LVO_RH = 1
            else:
                Diag_LVO_RH = 0

            ### 3. 최종 결과 (LH / RH / Bi / Normal)
            # - 좌/우반구 분류 결과 사용
            if (Diag_LVO_LH == 1) and (Diag_LVO_RH == 0):
                Diag_LVO = 'LH'
            elif (Diag_LVO_LH == 0) and (Diag_LVO_RH == 1):
                Diag_LVO = 'RH'
            elif (Diag_LVO_LH == 1) and (Diag_LVO_RH == 1):
                Diag_LVO = 'Bilateral'
            elif (Diag_LVO_LH == 0) and (Diag_LVO_RH == 0):
                Diag_LVO = 'Normal'
            else:
                logger_with_file.debug('Error - ELVO Decision Algorithm.')

        ### B. ED 결과가 양성인 경우
        #   -> ED 양성 방향 반구의 Prob만 분석
        else:
            ### 1. 타겟 반구 설정
            if (Diag_ED == 'LU') or (Diag_ED == 'LB'):
                target_hemi = 'LH'
            elif (Diag_ED == 'RU') or (Diag_ED == 'RB'):
                target_hemi = 'RH'
            else:
                logger_with_file.debug('Error - ELVO Decision Algorithm.')

            ### 2. 타겟 반구에 대해, decision
            if (result_LVO['Prob_LVO_{}'.format(target_hemi)] >= LVO_1st) and (
                    (result_LVO['Prob_EIC_{}'.format(target_hemi)] >= LVO_1st) or (
                    result_LVO['Prob_DMS_LH'] >= LVO_1st)):
                Diag_target_hemi = 1
            else:
                # 길병원 Triage 임상용 Threshold
                if inst_GIL:
                    # 2차
                    if (result_LVO['Prob_LVO_{}'.format(target_hemi)] >= LVO_2nd_Upper) and (
                            result_LVO['Prob_EIC_{}'.format(target_hemi)] >= LVO_2nd_Upper):
                        Diag_target_hemi = 1
                    elif (result_LVO['Prob_EIC_{}'.format(target_hemi)] >= LVO_2nd_Upper) and (
                            result_LVO['Prob_DMS_{}'.format(target_hemi)] >= LVO_2nd_Upper):
                        Diag_target_hemi = 1
                    elif (result_LVO['Prob_LVO_{}'.format(target_hemi)] >= LVO_2nd_Lower) and (
                            result_LVO['Prob_DMS_{}'.format(target_hemi)] >= LVO_2nd_DMS):
                        Diag_target_hemi = 1
                    else:
                        Diag_target_hemi = 0
                else:
                    # 2차
                    if (result_LVO['Prob_LVO_{}'.format(target_hemi)] >= LVO_2nd_Upper) and (
                            result_LVO['Prob_EIC_{}'.format(target_hemi)] >= LVO_2nd_Upper):
                        Diag_target_hemi = 1
                    elif (result_LVO['Prob_EIC_{}'.format(target_hemi)] >= LVO_2nd_Upper) and (
                            result_LVO['Prob_DMS_{}'.format(target_hemi)] >= LVO_2nd_Upper):
                        Diag_target_hemi = 1
                    elif (result_LVO['Prob_LVO_{}'.format(target_hemi)] >= LVO_2nd_Lower) and (
                            result_LVO['Prob_DMS_{}'.format(target_hemi)] >= LVO_2nd_Upper):
                        Diag_target_hemi = 1
                    else:
                        Diag_target_hemi = 0

            ### 3. 타겟 반구에 따른 최종 결과
            # - 반대 반구는 무조건 음성이므로, Bi-Lateral LVO 케이스는 없음
            if Diag_target_hemi == 1:
                if target_hemi == 'LH':
                    Diag_LVO = 'LH'
                elif target_hemi == 'RH':
                    Diag_LVO = 'RH'
                else:
                    logger_with_file.debug('Error - ELVO Decision Algorithm.')
            elif Diag_target_hemi == 0:
                Diag_LVO = 'Normal'
            else:
                logger_with_file.debug('Error - ELVO Decision Algorithm.')

        return Diag_LVO

    Diag_LVO = Diagnosis_LVO(result_LVO, Diag_ED)

    # 결과
    # -> ['LH', 'RH', 'Bilateral', 'Normal']
    return Diag_LVO


df = pd.read_csv(r'C:\Users\user\Downloads\test_val_prob.csv')
df_refine = pd.DataFrame(columns={'Patient_ID', 'Prob_LVO_LH', 'Prob_LVO_RH',
                                  'Prob_EIC_LH', 'Prob_EIC_RH',
                                  'Prob_DMS_LH', 'Prob_DMS_RH',
                                  'Prob_OI_LH', 'Prob_OI_RH',

                                  'ED_Cls_Left', 'ED_Cls_Right'
                                  })

for i in range(0, len(df), 2):
    cur_series = df.loc[i]
    cur_next_series = df.loc[i + 1]

    dict_return = {'Patient_ID': cur_series['Patient_ID'][:-2],
                   'Prob_LVO_LH': cur_series['Prob_LVO'],
                   'Prob_LVO_RH': cur_next_series['Prob_LVO'],
                   'Prob_EIC_LH': cur_series['Prob_EIC'],
                   'Prob_EIC_RH': cur_next_series['Prob_EIC'],
                   'Prob_DMS_LH': cur_series['Prob_DMS'],
                   'Prob_DMS_RH': cur_next_series['Prob_DMS'],
                   'Prob_OI_LH': cur_series['Prob_OI'],
                   'Prob_OI_RH': cur_next_series['Prob_OI'],

                   'ED_Cls_Left': cur_series['ED_Result'],
                   'ED_Cls_Right': cur_next_series['ED_Result']
                   }

    df_refine = df_refine.append(dict_return, ignore_index=True)

# df_refine = pd.read_csv(r'C:\Users\user\Downloads\길병원_triage_prob.csv')

df_refine = df_refine[['Patient_ID', 'Prob_LVO_LH', 'Prob_LVO_RH',
                       'Prob_EIC_LH', 'Prob_EIC_RH',
                       'Prob_DMS_LH', 'Prob_DMS_RH',
                       'Prob_OI_LH', 'Prob_OI_RH',

                       'ED_Cls_Left', 'ED_Cls_Right']]


# 환자 별로, DA 결과 뽑기
for i in range(0, len(df_refine)):
    cur_series = df_refine.loc[i]

    result_LVO = cur_series[['Patient_ID', 'Prob_LVO_LH', 'Prob_LVO_RH',
                             'Prob_EIC_LH', 'Prob_EIC_RH',
                             'Prob_DMS_LH', 'Prob_DMS_RH',
                             'Prob_OI_LH', 'Prob_OI_RH']].to_dict()
    result_ED = cur_series[['ED_Cls_Left', 'ED_Cls_Right']].to_dict()

    cur_result = DecisionAlgo_ELVO(result_LVO=result_LVO, result_ED=result_ED, inst_GIL=True)

    df_refine.loc[i, 'LVO_Result'] = cur_result

df_refine.to_excel('20221114_Result_LVO.xlsx', index=False)

