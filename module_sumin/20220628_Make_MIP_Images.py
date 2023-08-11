import os
import numpy as np
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
from PIL import Image
import matplotlib.pyplot as plt
import copy
import math
from knockknock import slack_sender

def createMIP(np_img, slices_num = 5):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amax(np_img[start:i+1],0)
    return np_mip

def createMIP_Sumin(np_img, slices_num=15):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-((slices_num-1)//2))
        end = i+((slices_num-1)//2)
        if end > img_shape[0]-1:
            end = img_shape[0]-1

        np_mip[i,:,:] = np.amax(np_img[start:end], 0)
    return np_mip

def createMIP_dicom_ALL(np_img):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    img_shape = np_img.shape
    np_mip = np.zeros([1, img_shape[1], img_shape[2]])

    start = 0
    end = img_shape[0]-1

    np_mip[0,:,:] = np.amax(np_img[start:end], 0)
    return np_mip


# DICOM windowing 수행 함수
def windowing(img, window_center, window_width, rescale=True):
    eps = 1e-10

    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    w_img = copy.deepcopy(img)

    w_img[w_img < window_min] = window_min
    w_img[w_img > window_max] = window_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        # w_img = (w_img - np.min(w_img)) / (np.max(w_img) - np.min(w_img) + eps)
        w_img = (w_img - window_min) / (window_max - window_min + eps)

    return w_img

path_root = "Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TestSET_Vessel" #hemi_original_WB_Vessel
# MIP 할 슬라이스 개수
slices_num_list = [5, 9, 13]
save_as_dicom = True
MIP_ALL = True

# for root_main, dirs_main, _ in os.walk(path_root):

# 환자 폴더 리스트
list_dir = sorted(os.listdir(path_root))

list_img = []




webhook_url = "https://hooks.slack.com/services/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"

@slack_sender(webhook_url=webhook_url, channel="#noti", user_mentions=["@정수민"])
def make_MIP():
    for cur_dir in list_dir:

        cur_target_dir = os.path.join(path_root, cur_dir, 'hemi_original_WB_Vessel')

        for root, dirs, files in os.walk(cur_target_dir):

            # Vessel Image Stack
            list_img = []

            for file in files:
                if file.endswith(".dcm"):
                    path = os.path.join(cur_target_dir, file)
                    ds = pydicom.dcmread(path)
                    img = ds.pixel_array

                    # windowing
                    # img = windowing(img, 40, 40)

                    # axis 추가 (shape 변경)
                    # img = np.expand_dims(img, axis=2)

                    list_img.append(img)

            img_stacked = np.stack((list_img), axis=0)

            # MIP ALL 아닌 경우에만, Slice num 기준으로 MIP
            if not MIP_ALL:

                # Slice Number 여러개에 대해서 생성
                for slices_num in slices_num_list:
                    img_MIP = createMIP_Sumin(img_stacked, slices_num=slices_num)

                    # print(img_MIP)

                    for i in range(0, len(files)):

                        cur_img_MIP = img_MIP[i]

                        # 해당 이미지에 0값 외의 픽셀이 없으면 저장하지 않음
                        cur_image_min = int(cur_img_MIP.min())
                        cur_image_max = int(cur_img_MIP.max())
                        if cur_image_min == 0 and cur_image_max == 0:
                            print('Error - {} _ {} 이미지는, 픽셀값이 모두 0'.format(cur_target_dir, files[i]))
                            continue
                        else:
                            plt.imshow(cur_img_MIP, cmap='gray')
                            # plt.show()
                            if save_as_dicom:
                                fn_to_save = files[i].replace('.dcm', '_Vessel_MIP_{}.dcm'.format(slices_num))
                                path_to_save = cur_target_dir.replace('hemi_original_WB_Vessel',
                                                                      'hemi_original_WB_Vessel_DCM_MIP_{}'.format(slices_num))
                                os.makedirs(path_to_save, exist_ok=True)

                                # path = os.path.join(cur_target_dir, file)
                                # ds = pydicom.dcmread(path)

                                # data = ds.pixel_array
                                # print('The image has {} x {} voxels'.format(data.shape[0],
                                #                                             data.shape[1]))
                                #
                                # print('The target image has {} x {} voxels'.format(
                                #     temp_img.astype(int).shape[0], temp_img.astype(int).shape[1]))

                                ds.PixelData = cur_img_MIP.astype(np.uint16).tobytes()

                                # update the information regarding the shape of the data array
                                # ds.Rows, ds.Columns = temp_img.astype(np.uint16).shape

                                ds.save_as(os.path.join(path_to_save, fn_to_save))
                            else:
                                fn_to_save = files[i].replace('.dcm', '_Vessel_MIP_{}.png'.format(slices_num))
                                path_to_save = cur_target_dir.replace('hemi_original_WB_Vessel', 'hemi_original_WB_Vessel_PNG_MIP_{}'.format(slices_num))
                                os.makedirs(path_to_save, exist_ok=True)

                                # plt.savefig(os.path.join(path_to_save, fn_to_save))
                                plt.imsave(os.path.join(path_to_save, fn_to_save), cur_img_MIP, cmap='gray')





            if MIP_ALL:
                # 모든 Slice를 MIP 1개로 만드는 방식
                img_MIP = createMIP_dicom_ALL(img_stacked)

                cur_img_MIP = img_MIP[0]

                # 픽셀값이 모두 0이면 저장 안함
                cur_image_min = int(cur_img_MIP.min())
                cur_image_max = int(cur_img_MIP.max())
                if cur_image_min == 0 and cur_image_max == 0:
                    print('Error - {} 이미지의 MIP_ALL은, 픽셀값이 모두 0'.format(cur_dir))
                    continue

                if save_as_dicom:
                    fn_to_save = cur_dir + '_Vessel_MIP_ALL.dcm'
                    path_to_save = cur_target_dir.replace('hemi_original_WB_Vessel',
                                                          'hemi_original_WB_Vessel_DCM_MIP_ALL')
                    os.makedirs(path_to_save, exist_ok=True)

                    ds.PixelData = cur_img_MIP.astype(np.uint16).tobytes()

                    print('test Print - {}'.format(os.path.join(path_to_save, fn_to_save)))
                    ds.save_as(os.path.join(path_to_save, fn_to_save))

                else:
                    fn_to_save = cur_dir + '_Vessel_MIP_ALL.png'
                    path_to_save = cur_target_dir.replace('hemi_original_WB_Vessel',
                                                          'hemi_original_WB_Vessel_PNG_MIP_ALL')
                    os.makedirs(path_to_save, exist_ok=True)

                    print('test Print - {}'.format(os.path.join(path_to_save, fn_to_save)))
                    plt.imsave(os.path.join(path_to_save, fn_to_save), cur_img_MIP, cmap='gray')


make_MIP()