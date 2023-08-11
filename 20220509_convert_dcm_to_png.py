from glob import glob
import copy
import pydicom
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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


path_dcm = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID★★★\ID_44cfc8e964'
path_result = r'./Result_dcm_to_png'
os.makedirs(path_result, exist_ok=True)

list_dcm = sorted(glob(path_dcm +'/*.dcm'))

obj_windowing = Windowing(WC=40, WW=80, rescale=True)


for path_cur_dcm in list_dcm:
    cur_dcm = pydicom.read_file(path_cur_dcm)
    cur_image = cur_dcm.pixel_array

    # rescale 수행
    intercept = -1024
    slope = 1
    cur_image = (cur_image * slope + intercept)


    cur_image = obj_windowing(cur_image)
    # cur_image = np.stack((cur_image,) * 3, axis=-1)

    cur_image = cur_image.astype(np.float64)

    plt.imshow(cur_image, cmap='gray')
    plt.imsave(os.path.join(path_result, os.path.basename(path_cur_dcm).replace('.dcm', '.png')), cur_image, cmap='gray')

    # png 저장