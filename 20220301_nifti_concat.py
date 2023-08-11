import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

# 1. Proxy 불러오기
data_LH = nib.load(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET\AJ0001\hemi_affined\hemi_L.nii.gz')
data_RH = nib.load(r'Z:\Stroke\cELVO\ELVO_Dataset_VoxelMorph\TrainSET\AJ0001\hemi_affined\hemi_R.nii.gz')

# 2. Header 불러오기
header = data_LH.header

# 3. 원하는 Header 불러오기 (내용이 문자열일 경우 숫자로 표현됨)
header_size = header['sizeof_hdr']

# 2. 전체 Image Array 불러오기
arr_LH = data_LH.get_fdata()
arr_RH = data_RH.get_fdata()

# 3. 원하는 Image Array 영역만 불러오기
sub_arr = data_LH.dataobj[..., 0:5]



# LH + RH 합연산 (같은 shape 일 때만 가능)
arr_sum = np.add(arr_LH, arr_RH) # or   'arr_LH + arr_RH'
arr_sum2 = arr_LH + arr_RH  # 위 합성이랑 같음
# arr_mean = np.add(arr_LH, arr_RH) / 2   # 이미지 옅어짐

# 합성된 arr 저장 (numpy -> nifti 로)
new_nifti = nib.Nifti1Image(arr_sum, affine=data_LH.affine)
nib.save(new_nifti, 'test.nii.gz')