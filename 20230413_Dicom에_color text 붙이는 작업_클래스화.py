# 2023.04.13
# - 목적
#   > dicom 파일 읽고
#   > 상단에, 어두운 여백 공간을 줄 것
#   > 여백 공간에 Color Text를 추가
#   > Color Text 추가 된 dicom 데이터를 저장

# - 헤딩 상세
#   > 진짜...쌩쑈를 했다.
#   > dicom 파일을 읽고, color text를 추가하고 싶었음
#   > overlay 하는 식으로 하려 했는데, pydicom 으로는 도저히 불가함
#       >> pydicom 에서, dicom.PixelData 로 접근하는 방식은 모두 불가함
#   > sitk 로 어찌어찌 저장함
#   > 단, 원본 데이터의 tag는 가져오면 저장이 아예 안되므로 생략함

import os
import numpy as np
import cv2
import SimpleITK as sitk
from typing import Tuple
import glob

class DicomModifier:
    def __init__(self, dicom_path):
        self.dicom_path = dicom_path

    def resize_dicom(self):
        # Read the DICOM file using SimpleITK
        dicom = sitk.ReadImage(self.dicom_path)
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

    def resize_dicom_and_add_black_space(self):
        # Resize the DICOM image using the provided function
        resized_dicom = self.resize_dicom()
        # Get the numpy array of the resized image data
        resized_npy = sitk.GetArrayFromImage(resized_dicom)
        # Create a black space of size (512, 100)
        black_space = np.full((100, 512), -1000, dtype=resized_npy.dtype)
        # Concatenate the black space and the resized image along the vertical axis
        combined_npy = np.vstack((black_space, resized_npy))
        combined_dicom = sitk.GetImageFromArray(combined_npy)

        # Copy metadata from the resized image
        for key in resized_dicom.GetMetaDataKeys():
            combined_dicom.SetMetaData(key, resized_dicom.GetMetaData(key))

        # Set the Photometric Interpretation to RGB
        combined_dicom.SetMetaData("0028|0004", "RGB")

        return combined_dicom

    def overlay_text_color_sitk(self, text, save_path, resized_dicom, position_ratio=(0.1, 0.9), text_thickness=10):
        # Read the original DICOM file using SimpleITK
        org = resized_dicom
        # Convert the image to a floating point format
        img_float = sitk.Cast(org, sitk.sitkFloat32)

        # Apply windowing to the image data
        window_center = org.GetMetaData("0028|1050")
        window_width = org.GetMetaData("0028|1051")
        img_window = sitk.IntensityWindowing(img_float, windowMinimum=float(window_center) - 0.5 * float(window_width),
                                             windowMaximum=float(window_center) + 0.5 * float(window_width), outputMinimum=0.0,
                                             outputMaximum=255.0)

        # Rescale the pixel values to uint8
        img_uint8 = sitk.Cast(sitk.RescaleIntensity(img_window), sitk.sitkUInt8)
        # Get the numpy array of the image data
        npy = sitk.GetArrayFromImage(img_uint8)

        if len(npy.shape) == 2:
            img_3channel = cv2.cvtColor(npy, cv2.COLOR_GRAY2RGB)
        else:
            img_3channel = npy

        mask = np.zeros_like(img_3channel)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 138, 138)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img_3channel.shape[1] - text_size[0]) // 2
        text_y = 70 # y축 위치 수정
        text_position = (text_x, text_y)
        cv2.putText(img_3channel, text, text_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
        text_img = cv2.addWeighted(img_3channel, 1, mask, 1, 0)
        mod = sitk.GetImageFromArray(np.expand_dims(text_img, axis=0))

        if os.path.exists(save_path):
            os.remove(save_path)

        sitk.WriteImage(mod, save_path)

# input_dicom = r"D:\OneDrive\00000000_Code\20221102_cSTROKE\0_engine_version\cSTROKE\git_dev_internal\Data_Sample\PreProcessed\NCCT_sort\HeuronStroke_006.dcm"
path_dir = r"D:\OneDrive\00000000_Code\20221102_cSTROKE\0_engine_version\cSTROKE\git_dev_internal\Data_Sample\PreProcessed\NCCT_sort"
text = 'Suspected Hemorrhage'

list_dicom = glob.glob(os.path.join(path_dir, '*.dcm'))
for path_dicom in list_dicom:
    output_dicom = os.path.basename(path_dicom).replace('.dcm', '_Series.dcm')

    dicom_mod = DicomModifier(path_dicom)
    resized_dicom = dicom_mod.resize_dicom_and_add_black_space()
    dicom_mod.overlay_text_color_sitk(text=text, save_path=output_dicom, resized_dicom=resized_dicom)