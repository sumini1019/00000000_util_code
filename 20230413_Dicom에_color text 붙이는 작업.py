# 2023.04.13
# - 목적
#   > dicom 파일 읽고
#   > 상단에, 어두운 여백 공간을 줄 것
#   > 여백 공간에 Color Text를 추가
#   > Color Text 추가 된 dicom 데이터를 저장
#
# - 헤딩 상세
#   > 진짜...쌩쑈를 했다.
#   > dicom 파일을 읽고, color text를 추가하고 싶었음
#   > overlay 하는 식으로 하려 했는데, pydicom 으로는 도저히 불가함
#       >> pydicom 에서, dicom.PixelData 로 접근하는 방식은 모두 불가함
#   > sitk 로 어찌어찌 저장함
#   > 단, 원본 데이터의 tag는 가져오면 저장이 아예 안되므로 생략함

import os.path
from typing import Tuple
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import shutil

# def window_image_with_rescale(image, window_center, window_width, rescale_slope, rescale_intercept):
#     img = image * rescale_slope + rescale_intercept
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     window_image = img.copy()
#     window_image[window_image < img_min] = img_min
#     window_image[window_image > img_max] = img_max
#     return window_image

# def overlay_text(
#                  org_path:str,
#                  text:str,
#                  save_path:str,
#                  position_ratio:Tuple[float, float]= (0.1, 0.9),
#                  text_thickness:int = 10
#                 ) -> None:
#     org = pydicom.dcmread(org_path)
#     npy = org.pixel_array
#     position = (int(npy.shape[1]*position_ratio[0]),int(npy.shape[0]*position_ratio[1]))
#     dummy = np.zeros(npy.shape)
#     text_img = cv2.putText(dummy, text, position, 0, 3, (255,255,255), text_thickness)
#     # text_img = cv2.putText(dummy, text, position, 0, 3, (255, 0, 0), text_thickness)
#     text_img = ((text_img / 255) * npy.max()).astype(npy.dtype)
#     overlay = np.maximum(text_img, npy)
#     mod = pydicom.dcmread(org_path)
#     mod.PixelData = overlay.tobytes()
#     mod.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
#     mod.save_as(save_path)

# def overlay_text_color(
#                  org_path:str,
#                  text:str,
#                  save_path:str,
#                  position_ratio:Tuple[float, float]= (0.1, 0.9),
#                  text_thickness:int = 10
#                 ) -> None:
#     org = pydicom.dcmread(org_path)
#     npy = org.pixel_array
#
#     # npy = convert_to_8bit(npy)
#
#     if org.PhotometricInterpretation.startswith('MONOCHROME'):
#         img_3channel = cv2.cvtColor(npy, cv2.COLOR_GRAY2BGR)
#     else:
#         img_3channel = npy
#
#     position = (int(img_3channel.shape[1] * position_ratio[0]), int(img_3channel.shape[0] * position_ratio[1]))
#
#     mask = np.zeros_like(img_3channel)
#     cv2.putText(mask, text, position, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), text_thickness)
#
#     text_img = cv2.addWeighted(img_3channel, 1, mask, 1, 0)
#
#     mod = pydicom.dcmread(org_path)
#
#     mod.SamplesPerPixel = 3
#     mod.PlanarConfiguration = 0
#     mod.Rows, mod.Columns, _ = text_img.shape
#
#     mod.PhotometricInterpretation = "RGB"
#     mod.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
#
#     mod.PixelData = text_img.tobytes()
#
#     mod.save_as(save_path)

# def convert_to_8bit(image: np.ndarray) -> np.ndarray:
#     min_val = np.min(image)
#     max_val = np.max(image)
#     image_8bit = (image - min_val) / (max_val - min_val) * 255
#     return image_8bit.astype(np.uint8)

def resize_dicom(dicom_path):
    # Read the DICOM file using SimpleITK
    dicom = sitk.ReadImage(dicom_path)
    # Get the numpy array of the image data
    npy = sitk.GetArrayFromImage(dicom)[0]
    # Resize the image to (512, 512)
    resized_npy = cv2.resize(npy, (512, 512))
    # Convert the resized numpy array back to SimpleITK image
    resized_dicom = sitk.GetImageFromArray(resized_npy)
    # Copy metadata from the original image
    for key in dicom.GetMetaDataKeys():
        resized_dicom.SetMetaData(key, dicom.GetMetaData(key))

    # Set the Photometric Interpretation to RGB
    resized_dicom.SetMetaData("0028|0004", "RGB")

    return resized_dicom

def resize_dicom_and_add_black_space(dicom_path):
    # Resize the DICOM image using the provided function
    resized_dicom = resize_dicom(dicom_path)

    # Get the numpy array of the resized image data
    resized_npy = sitk.GetArrayFromImage(resized_dicom)

    # Create a black space of size (512, 100)
    black_space = np.full((100, 512), -1000, dtype=resized_npy.dtype)

    # Concatenate the black space and the resized image along the vertical axis
    combined_npy = np.vstack((black_space, resized_npy))

    # Convert the combined numpy array back to SimpleITK image
    combined_dicom = sitk.GetImageFromArray(combined_npy)

    # Copy metadata from the resized image
    for key in resized_dicom.GetMetaDataKeys():
        combined_dicom.SetMetaData(key, resized_dicom.GetMetaData(key))

    # Set the Photometric Interpretation to RGB
    combined_dicom.SetMetaData("0028|0004", "RGB")

    return combined_dicom




def overlay_text_color_sitk(resized_dicom, mode='', direction_LVO='') -> None:
    # Read the original DICOM file using SimpleITK
    org = resized_dicom

    # Convert the image to a floating point format
    img_float = sitk.Cast(org, sitk.sitkFloat32)

    # Apply windowing to the image data
    window_center = org.GetMetaData("0028|1050")
    window_width = org.GetMetaData("0028|1051")
    img_window = sitk.IntensityWindowing(img_float, windowMinimum=float(window_center)-0.5*float(window_width), windowMaximum=float(window_center)+0.5*float(window_width), outputMinimum=0.0, outputMaximum=255.0)

    # Rescale the pixel values to uint8
    img_uint8 = sitk.Cast(sitk.RescaleIntensity(img_window), sitk.sitkUInt8)

    # Get the numpy array of the image data
    npy = sitk.GetArrayFromImage(img_uint8) #[0]

    if len(npy.shape) == 2:
        img_3channel = cv2.cvtColor(npy, cv2.COLOR_GRAY2RGB)
    else:
        img_3channel = npy

    mask = np.zeros_like(img_3channel)

    # Font / 색상 설정
    if mode=='ICH_P':
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 138, 138)

        text = 'Suspected Hemorrhage'
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img_3channel.shape[1] - text_size[0]) // 2
        text_y = 70  # y축 위치 수정
        text_position = (text_x, text_y)

        cv2.putText(img_3channel, text, text_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    elif mode=='ELVO_P':
        # 1번 텍스트 추가
        font1 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale1 = 0.55  # 1번 텍스트용 작은 폰트 크기
        color1 = (224, 224, 224)  # 1번 텍스트용 회색
        thickness1 = 1

        text1 = "Large vessel occlusion is"
        text_size1 = cv2.getTextSize(text1, font1, font_scale1, thickness1)[0]
        text_x1 = (img_3channel.shape[1] - text_size1[0]) // 2
        text_y1 = 60  # 2번 텍스트보다 위에 위치
        text_position1 = (text_x1, text_y1)
        cv2.putText(img_3channel, text1, text_position1, font1, font_scale1, color1, thickness1, lineType=cv2.LINE_AA)


        # 2번 텍스트 추가
        font2 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale2 = 0.8
        thickness2 = 2
        color2 = (255, 138, 138)

        text2 = f"Suspected ({direction_LVO})"
        text_size2 = cv2.getTextSize(text2, font2, font_scale2, thickness2)[0]
        text_x2 = (img_3channel.shape[1] - text_size2[0]) // 2
        text_y2 = 90  # 1번 텍스트보다 아래에 위치
        text_position2 = (text_x2, text_y2)
        cv2.putText(img_3channel, text2, text_position2, font2, font_scale2, color2, thickness2, lineType=cv2.LINE_AA)

    elif mode=='ELVO_N':
        # 1번 텍스트 추가
        font1 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale1 = 0.55  # 1번 텍스트용 작은 폰트 크기
        color1 = (224, 224, 224)  # 1번 텍스트용 회색
        thickness1 = 1

        text1 = "Large vessel occlusion is"
        text_size1 = cv2.getTextSize(text1, font1, font_scale1, thickness1)[0]
        text_x1 = (img_3channel.shape[1] - text_size1[0]) // 2
        text_y1 = 60  # 2번 텍스트보다 위에 위치
        text_position1 = (text_x1, text_y1)
        cv2.putText(img_3channel, text1, text_position1, font1, font_scale1, color1, thickness1, lineType=cv2.LINE_AA)

        # 2번 텍스트 추가
        font2 = cv2.FONT_HERSHEY_SIMPLEX
        font_scale2 = 0.8
        thickness2 = 2
        color2 =  (130, 219, 237)

        text2 = f"Not Suspected"
        text_size2 = cv2.getTextSize(text2, font2, font_scale2, thickness2)[0]
        text_x2 = (img_3channel.shape[1] - text_size2[0]) // 2
        text_y2 = 90  # 1번 텍스트보다 아래에 위치
        text_position2 = (text_x2, text_y2)
        cv2.putText(img_3channel, text2, text_position2, font2, font_scale2, color2, thickness2, lineType=cv2.LINE_AA)
    else:
        print('Error')
        return False


    text_img = cv2.addWeighted(img_3channel, 1, mask, 1, 0)

    # Convert the modified image back to SimpleITK image
    mod = sitk.GetImageFromArray(np.expand_dims(text_img, axis=0))

    # # Copy metadata from the original image
    # for key in org.GetMetaDataKeys():
    #     mod.SetMetaData(key, org.GetMetaData(key))
    # # Set the Photometric Interpretation to RGB
    # mod.SetMetaData("0028|0004", "RGB")

    return mod



# input_dicom = r"D:\OneDrive\00000000_Code\20221102_cSTROKE\0_engine_version\cSTROKE\git_dev_internal\Data_Sample\PreProcessed\NCCT_sort\HeuronStroke_006.dcm"

mode = 'ELVO_N' #'ICH_P', 'ELVO_P', 'ELVO_N'
direction_LVO = 'L'
path_input_dir = r"D:\OneDrive\00000000_Code\20221102_cSTROKE\0_engine_version\cSTROKE\git_dev_internal\Data_Sample\PreProcessed\NCCT_sort"
path_output_dir = r'D:\OneDrive\00000000_Code\00000000_util_code\temp'



# if os.path.exists(path_output_dir):
#     shutil.rmtree(path_output_dir)
# os.makedirs(path_output_dir)

list_dicom = glob.glob(os.path.join(path_input_dir, '*.dcm'))
for path_input in list_dicom:
    path_output = os.path.join(path_output_dir, os.path.basename(path_input).replace('.dcm', f'_Series_{mode}.dcm'))

    resized_dicom = resize_dicom_and_add_black_space(path_input)
    modified_dicom = overlay_text_color_sitk(resized_dicom=resized_dicom, mode=mode, direction_LVO=direction_LVO)

    sitk.WriteImage(modified_dicom, path_output)