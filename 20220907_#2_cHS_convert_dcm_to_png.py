# 2022.09.07
# - 폴더 별 Dicom 파일을, PNG 변환 후 저장

from glob import glob
import copy
import pydicom
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class Windowing(object):
    def __init__(self, WC=40, WW=80, rescale=True):
        self.WC = WC
        self.WW = WW
        self.rescale = rescale
    def __call__(self, image):
        w_image = windowing(image, self.WC, self.WW, self.rescale)
        #return torch.from_numpy(w_image).to(torch.float32).permute(0,3,1,2) # 1, H, W, D >> 1, D, H, W
        # return torch.from_numpy(w_image).to(torch.float32)
        return w_image

def windowing(img, window_center, window_width, rescale=True):
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    w_img = copy.deepcopy(img)

    w_img[w_img < window_min] = window_min
    w_img[w_img > window_max] = window_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        w_img = (w_img - window_min) / (window_max - window_min)



    return w_img

# 원본 경로
# path_source = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\1. 2D Annotation\1. Data_Series\Hemo'
path_source = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\1. 2D Annotation\1. Data_Series\Normal'
# 복사할 경로
# path_dest = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\1. 2D Annotation\2. Data_Series (png, label)\Hemo'
path_dest = r'H:\20220907_탑병원 데이터, Annotation 프로그램 전달\1. 2D Annotation\2. Data_Series (png, label)\Normal'


from knockknock import slack_sender
webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="#noti", user_mentions=["@정수민"])
def main():
    # Windowing 오브젝트 생성
    obj_windowing = Windowing(WC=40, WW=80, rescale=True)

    # 원본의 Series 별로 변환 / 복사 진행
    list_series = os.listdir(path_source)

    # Series의 DataFrame 생성
    # - DCM Axial 축으로 정렬 목적

    for cur_series in list_series:
        try:
            # 현재 series의 경로
            path_cur_series = os.path.join(path_source, cur_series)

            # 현재 series의 dcm 리스트
            list_dcm = sorted(glob(path_cur_series + '/*.dcm'))

            # Series의 DataFrame 생성
            # - DCM Axial 축으로 정렬 목적
            df_series = pd.DataFrame(columns={"Slice_ID", "Position"})

            # DCM 로드 후, Position 확인 후, DataFrame 정보만 삽입
            for path_cur_dcm in list_dcm:
                # 현재 dcm의 파일명
                cur_fn = os.path.basename(path_cur_dcm).replace('.dcm', '')
                # dcm 열기 / Array 변환
                cur_dcm = pydicom.read_file(path_cur_dcm)

                # DataFrame에 Input
                dict_result = {"Slice_ID": path_cur_dcm, "Position": cur_dcm.ImagePositionPatient[2]}
                df_series = df_series.append(dict_result, ignore_index=True)

            # DataFrame 정렬 (Axial 순서에 따라)
            df_series = df_series.sort_values('Position', ascending=True).reset_index(drop=True)
            # Axial 순서에 맞게, 데이터 리스트 생성
            list_dcm = list(df_series['Slice_ID'])

            # 현재 Series의 모든 dcm 변환 및 복사
            # for path_cur_dcm in list_dcm:
            # for path_cur_dcm in list(df_series['Slice_ID']):
            for i in range(0, len(list_dcm)):
                # 현재 dcm의 파일명
                cur_fn = os.path.basename(list_dcm[i]).replace('.dcm', '')
                # 복사할 경로
                if i < 10:
                    path_dest_cur_png = os.path.join(path_dest, cur_series,
                                                     'SE{}_00{}_SL'.format(cur_series, str(i)) + cur_fn + '.png')
                elif 10 <= i < 100:
                    path_dest_cur_png = os.path.join(path_dest, cur_series,
                                                     'SE{}_0{}_SL'.format(cur_series, str(i)) + cur_fn + '.png')
                else:
                    path_dest_cur_png = os.path.join(path_dest, cur_series,
                                                     'SE{}_{}_SL'.format(cur_series, str(i)) + cur_fn + '.png')

                # dcm 열기 / Array 변환
                cur_dcm = pydicom.read_file(list_dcm[i])
                cur_image = cur_dcm.pixel_array
                # rescale 수행
                intercept = -1024
                slope = 1
                cur_image = (cur_image * slope + intercept)

                # Windowing
                cur_image = obj_windowing(cur_image)
                # cur_image = np.stack((cur_image,) * 3, axis=-1)

                # type 변환
                cur_image = cur_image.astype(np.float64)

                # 이미지 저장
                os.makedirs(os.path.join(path_dest, cur_series), exist_ok=True)
                plt.imsave(path_dest_cur_png, cur_image, cmap='gray')

        except:
            print('Error - DCM to PNG 파일 변환 실패 ({})'.format(cur_series))

main()