import glob
import os
import SimpleITK as sitk

mode_train = 'train' # 'test'
mode = 'label' #'image', 'label'

# 2023.06.30
# - Chronic은 Hemorrhage Label에서 제외
exclude_ChronicSDH = True

if mode == 'image':
    # path_src = fr'D:\00000000 Code\20230523_SwinUNETR_ICH\data_ICH\{mode_train}\image_Hemo_nifti'
    # path_dst = fr'D:\00000000 Code\20230523_SwinUNETR_ICH\data_ICH\{mode_train}\image_Hemo_nifti_Resize_256'
    path_src = fr'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
    path_dst = fr'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)_Resize_256'

    list_data = glob.glob(path_src + '/*.nii.gz')
else:
    # path_src = fr'D:\00000000 Code\20230523_SwinUNETR_ICH\data_ICH\{mode_train}\label_nrrd(until_5th)'
    # path_dst = fr'D:\00000000 Code\20230523_SwinUNETR_ICH\data_ICH\{mode_train}\label_nrrd(until_5th)_Resize_256'
    path_src = fr'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_7th)'
    path_dst = fr'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_7th)_Resize_256'

    if exclude_ChronicSDH:
        path_dst = path_dst + '_exclude_ChronicSDH'

    list_data = glob.glob(path_src + '/*.nrrd')

os.makedirs(path_dst, exist_ok=True)

# 원하는 크기 지정
new_size = [256, 256, 256]

for item in list_data:
    # 이미지 불러오기
    image = sitk.ReadImage(item)

    # 원본 이미지 메타데이터 저장
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()

    # Label일 경우, 모든 Label 값을 1로 변환
    if mode == 'label':
        # 이미지를 numpy 배열로 변환
        array = sitk.GetArrayFromImage(image)

        # 2023.06.30
        # - Chronic Hemorrhage는 제외
        if exclude_ChronicSDH:
            array[array == 6] = 0

        # 배열에서 1 이상인 값을 모두 1로 변경
        array[array >= 1] = 1


        # numpy 배열을 다시 이미지로 변환
        image = sitk.GetImageFromArray(array)

        # 원본 이미지 메타데이터 복원
        image.SetDirection(original_direction)
        image.SetOrigin(original_origin)
        image.SetSpacing(original_spacing)
    # # Image일 경우, Smoothing 적용
    # elif mode == 'image':
    #     # Gaussian smoothing 필터 적용
    #     smoothingFilter = sitk.SmoothingRecursiveGaussianImageFilter()
    #     smoothingFilter.SetSigma(1.0)  # Sigma 값 조정으로 smoothing의 정도를 조절
    #     image = smoothingFilter.Execute(image)

    # 새로운 간격 계산
    new_spacing = [osz*osp/nsp for osz, osp, nsp in zip(original_size, original_spacing, new_size)]

    # 리사이징 필터 적용
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    # resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetDefaultPixelValue(0)  # 여기를 수정
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # (sitk.sitkLinear)  # 보간법 설정


    # 리사이징 이미지 생성
    resized_image = resampler.Execute(image)

    # 이미지 저장
    # sitk.WriteImage(resized_image, os.path.join(path_dst, f'{os.path.basename(item)}.nii.gz'))
    sitk.WriteImage(resized_image, os.path.join(path_dst, f'{os.path.basename(item)}'))
