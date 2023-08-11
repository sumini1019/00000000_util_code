# 2022.08.30
# - 기존 Nifti 변환 코드 기반으로 변형
# - cHS Series 데이터에 대해서, Nifti 변환

# sitk 1.2.4
import os
import SimpleITK as sitk
import numpy as np
import pydicom
from PIL import Image
from glob import glob

def get_images_from_directory(path):
    try:
        series_ID = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_ID[0])
    except:
        series_file_names = glob(path + '/*.dcm')
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image = series_reader.Execute()
    return image

# 2022.08.30
# - RSNA 데이터의 경우, Series UID 정보가 기존 데이터와 다름
# - Series UID를 Dicom에서 읽어들이지 않고, 그대로 파일 리스트 Tuple 만들도록 변경
def get_images_from_directory_FOR_RSNA(path):
    # series_ID = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    # print('Series_ID (Image) : ', series_ID)
    # series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_ID[0])
    series_file_names = tuple(os.listdir(path))
    for i in range(0, len(series_file_names)):
        series_file_names[i] = os.path.join(path, series_file_names[i])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()

    # 여기서 에러 난다ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ
    image = series_reader.Execute()
    return image

def convert_to_sitk(array_image, spacing, origin, direction=None):
    itk_image = sitk.GetImageFromArray(array_image)
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)
    if direction:
        itk_image.SetDirection(direction)
    return itk_image

def main():
    # Nifty 변환 할, Image 폴더
    # image_directory = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID'
    image_directory = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210119_cHS_Gil_Data\Original\Hemo_n38\dicom'

    list_patient_folder = os.listdir(image_directory)

    # Mask png 파일의 위치
    mask_path = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\mask'
    # Mask png 파일에 대응하는 DCM
    mask_dcm_path = r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\dcm_by_id'
    # Mask 파일이 존재하는, Patient ID 리스트
    list_id_mask_dcm = os.listdir(mask_dcm_path)

    for cur_patient in list_patient_folder:
        cur_image_folder = os.path.join(image_directory, cur_patient)

        list_image = os.listdir(cur_image_folder)

        image = get_images_from_directory(cur_image_folder)
        # image = get_images_from_directory_FOR_RSNA(cur_image_folder)

        # Nifty (dcm 변환) 파일의, 데이터의 공간정보 받아오기
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        print(image.GetSize()) # (512, 512, 40)  40 : slice 개수

        # Nifty (dcm 변환) 파일을, numpy 형태로 변환
        array_image = sitk.GetArrayFromImage(image)
        print(array_image.shape) # (40, 512, 512)   40 : slice 개수

        ###################################
        ### Mask 파일을, Nifty 형태로 변환 ###
        ###################################

        # 현재 Image에 해당하는, 마스크 파일이 있는가?
        # - 있다면, Mask 파일을 Nifty 파일 형태로 생성
        if cur_patient in list_id_mask_dcm:
            # Mask dcm 파일들 파싱
            list_mask_dcm = os.listdir(os.path.join(mask_dcm_path, cur_patient))

            series_ID_mask_dcm = sitk.ImageSeriesReader.GetGDCMSeriesIDs(mask_dcm_path)
            print('Series_ID (Mask) : ', series_ID_mask_dcm)

            # 1. Mask png 파일을, Numpy Array 로 변환하여 객체화
            #   -> *** 반드시, DCM과 동일한 순서대로 Numpy 채널 순서를 맞춰야 함
            #   -> *** Mask가 없는 Slice는, 그냥 zeros() 로 채워진 numpy array 삽입

            # array_mask = []#np.empty()
            cnt_array = 0

            # 1-1. Image 와 동일한 Slice 개수만큼 반복
            for idx in range(0, array_image.shape[0]):
                # Mask DCM 중, 같은 Image Position인 Slice가 있는지 검사
                # 1. Image의 Position
                cur_image = pydicom.read_file(os.path.join(cur_image_folder, list_image[idx]))
                cur_image_position = cur_image.ImagePositionPatient
                # 2. Mask의 Position 리스트 파싱
                list_mask_dcm_position = []
                for cur_mask_path in list_mask_dcm:
                    cur_mask = pydicom.read_file(os.path.join(mask_dcm_path, cur_patient, cur_mask_path))
                    list_mask_dcm_position.append(cur_mask.ImagePositionPatient)
                # 3. Mask의 ImagePosition 중, 같은게 있는지?
                # 3-1. 있다면 -> 해당 Slice를 읽고, ndarray 변환 후, 전체 ndarray에 누적
                if cur_image_position in list_mask_dcm_position:
                    # print('exist mask')
                    # print('Index : ', list_mask_dcm_position.index(cur_image_position))
                    mask_idx = list_mask_dcm_position.index(cur_image_position)
                    name_mask = list_mask_dcm[mask_idx]
                    name_mask = name_mask.replace('.dcm', '_label.png')
                    # Mask Png 파일 로드 후 ndarray 변환
                    mask = Image.open(os.path.join(mask_path, name_mask))
                    np_mask = np.array(mask)
                    # DMS 제외한, Label은 제거 (2, 3 값은 모두 0으로 치환)
                    np_mask = np.where(np_mask > 1.0, 0, mask)
                    # Stack 가능하도록, shape 변경
                    np_mask = np_mask.reshape(1, 512, 512)

                # 3-2. 없다면 -> 0 으로만 채워진 array를 누적
                else:
                    # print('not exist mask')
                    # stacked_array.append(np.zeros(shape=(1, array_image.shape[1], array_image.shape[2])))
                    np_mask = np.zeros(shape=(1, array_image.shape[1], array_image.shape[2]))

                # Numpy Array 누적
                if cnt_array == 0:
                    stacked_array = np_mask.reshape(1, 512, 512)
                else:
                    stacked_array = np.vstack((stacked_array, np_mask))

                cnt_array = cnt_array + 1

            # 2. Mask와, DCM Nifty 변환한 array의 shape와 동일한지 확인
            print(stacked_array.shape)  # (40, 512, 512)    40 : slice 개수

            # 3. 동일하다면, Mask도 Nitfy 변환
            mask = convert_to_sitk(stacked_array, spacing, origin, direction)

            # 4. Nifty 저장 (dicom 이미지, mask 이미지)
            sitk.WriteImage(image, r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\nifti\{}_image.nii.gz'.format(cur_patient))
            sitk.WriteImage(mask, r'Z:\Sumin_Jung\00000000_DATA\3_cELVO\20201123_DMS_annotation(dms,calcification)\WithS\nifti\{}_mask.nii.gz'.format(cur_patient))

        # - 없다면, Mask 파일 생성 X
        else:
            print('No Mask File : Patient {}', cur_patient)

    #########
    # 3D Slicer에, Nifty 넣어서 확인해 볼 것

if __name__ == "__main__":
    main()