import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
import cv2
import nrrd
import shutil

def windowing(img, wc, ww):
    min_val = wc - ww / 2
    max_val = wc + ww / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / ww * 255
    img = img.astype(np.uint8)
    return img

def split_by_seriesID(path_img, path_img_split):
    # path_img 폴더에 있는 파일들을 읽어들임
    for filename in os.listdir(path_img):
        try:
            # 파일명에서 첫 네 자리를 추출하여 ID 값을 얻음
            img_id = filename.split("_idx")[0]

            # path_img_split/ID 폴더를 생성
            dst_dir = os.path.join(path_img_split, img_id)
            os.makedirs(dst_dir, exist_ok=True)

            # 파일을 path_img_split/ID 폴더로 복사
            src_file = os.path.join(path_img, filename)
            dst_file = os.path.join(dst_dir, filename)
            shutil.copyfile(src_file, dst_file)
        except:
            print(f'파일 복사 실패 - {filename}')


# 2023.04.25
# - have_label 파라미터 추가
#   > True  : 대응하는 Label 데이터 있음. 이미지,label 변환 작업 수행
#   > False : 대응하는 Label 데이터 없음. 이미지 데이터만 변환
def save_2d_slices(image_path, label_path,
                   path_output_img, path_output_label_subtype, path_output_label_binary,
                   path_output_label_binary_exclude_Chronic, path_output_label_3class,
                   have_label=True):
    os.makedirs(path_output_img, exist_ok=True)
    if have_label:
        os.makedirs(path_output_label_subtype, exist_ok=True)
        os.makedirs(path_output_label_binary, exist_ok=True)
        os.makedirs(path_output_label_binary_exclude_Chronic, exist_ok=True)
        os.makedirs(path_output_label_3class, exist_ok=True)

    # Load image and label data
    image_sitk = sitk.ReadImage(image_path)

    # Modify the origin
    origin = np.array(image_sitk.GetOrigin())
    origin[0] *= -1  # modify x-coordinate
    origin[1] *= -1  # modify y-coordinate
    image_sitk.SetOrigin(tuple(origin))

    if have_label:
        label, label_header = nrrd.read(label_path)

    # Get image and label numpy arrays
    image_np = sitk.GetArrayFromImage(image_sitk)
    if have_label:
        label_np = np.transpose(label, (2, 1, 0))

    # Define label colormap
    label_colormap = {
        0: (0, 0, 0),  # background: black
        1: (255, 0, 0),  # label 1: red
        2: (0, 255, 0),  # label 2: green
        3: (0, 0, 255),  # label 3: blue
        4: (255, 255, 0),  # label 4: yellow
        5: (255, 0, 255),  # label 5: magenta
        6: (0, 255, 255),  # label 6: cyan
        7: (128, 0, 0),  # label 7: maroon
        8: (0, 128, 0),  # label 8: green (128)
        9: (0, 0, 128),  # label 9: navy
        10: (128, 128, 128)  # label 10: gray
        # add more colors for additional labels as needed
    }

    # Save 2D PNG files with windowing
    for i in range(image_np.shape[0]):
        # Apply windowing
        image_windowed = windowing(image_np[i], wc=40, ww=80)

        # Save PNG image
        image_file = os.path.basename(image_path)
        image_name, _ = os.path.splitext(image_file)
        image_name, _ = os.path.splitext(image_name)
        image_name_with_index = f"{image_name}_idx_{i:03d}"
        image_name_with_index_png = f"{image_name_with_index}.png"
        image_file_path = os.path.join(path_output_img, image_name_with_index_png)
        cv2.imwrite(image_file_path, image_windowed)


        # Save label
        if have_label:
            label_slice = label_np[i]  # i번째 인덱스에 해당하는 라벨 데이터 슬라이스 가져오기
            label_img = Image.fromarray(label_slice.astype(np.uint8), mode='P')  # NumPy 배열로부터 PIL 이미지 객체 생성
            label_img.putpalette([color for value in sorted(label_colormap.keys()) for color in label_colormap[value]]) # 라벨 컬러맵을 사용하여 이미지의 팔레트 설정
            # label_img.save(os.path.join(path_output_label, f'{os.path.splitext(os.path.basename(label_path))[0]}_idx_{i:03d}.png'))

            # Save label binary (with pixel value 1 for label 1 or higher)
            label_binary_slice = np.zeros_like(label_slice)  # 라벨 이진화 슬라이스 생성
            label_binary_slice[label_slice >= 1] = 1  # 라벨 값이 1 이상인 픽셀에 대해 이진화 슬라이스 설정
            label_binary_img = Image.fromarray(label_binary_slice.astype(np.uint8), mode='P')  # 이진화 슬라이스로부터 PIL 이미지 객체 생성
            # label_binary_img.putpalette(bytes([0, 0, 0, 255, 0, 0]))  # 이진화 이미지의 팔레트 설정
            label_binary_img.putpalette([color for value in sorted(label_colormap.keys()) for color in label_colormap[value]])

            # 2023.07.19
            # - Chronic을 제외한 나머지 Hemorrhage만 1 값으로 변경하는 Binary Label 이미지 생성
            # (0: BG + Chronic, 1: Hemo_exclude Chronic)
            label_binary_slice_exclude_Chronic = np.zeros_like(label_slice)
            # 라벨 값이 1 이상이면서 6이 아닌 픽셀에 대해 이진화 슬라이스를 1로 설정
            label_binary_slice_exclude_Chronic[(label_slice >= 1) & (label_slice != 6)] = 1
            # 라벨 값이 6인 픽셀에 대해 이진화 슬라이스를 0으로 설정
            label_binary_slice_exclude_Chronic[label_slice == 6] = 0
            # 이진화 슬라이스로부터 PIL 이미지 객체 생성
            label_binary_img_exclude_Chronic = Image.fromarray(label_binary_slice_exclude_Chronic.astype(np.uint8), mode='P')
            # 이진화 이미지의 팔레트 설정 (원래 코드에 맞게 조정하세요)
            label_binary_img_exclude_Chronic.putpalette([color for value in sorted(label_colormap.keys()) for color in label_colormap[value]])

            # 3 Class 라벨 생성 (0: BG, 1: Hemo, 2: Chronic)
            # Create label_3class object
            label_3class = np.zeros_like(label_slice)  # label_slice와 동일한 크기의 배열을 모두 0으로 초기화
            # Set pixel value 0 for label value 0
            label_3class[label_slice == 0] = 0
            # Set pixel value 2 for label value 6 (Chronic 클래스)
            label_3class[label_slice == 6] = 2
            # Set pixel value 1 for label values other than 0, 6 (일반 출혈)
            label_3class[np.logical_and(label_slice != 0, label_slice != 6)] = 1
            # 팔레트적용 + PIL 변환
            label_3class_img = Image.fromarray(label_3class.astype(np.uint8), mode='P')  # 이진화 슬라이스로부터 PIL 이미지 객체 생성
            label_3class_img.putpalette([color for value in sorted(label_colormap.keys()) for color in label_colormap[value]])

            # Check if pixel value is 8 or higher
            if np.any(label_slice >= 8):
                print(f"Error: Pixel value 8 or higher encountered! : {image_path}")
                return False

            label_name = os.path.splitext(os.path.basename(label_path))[0]  # 라벨 파일의 이름 가져오기
            label_name_with_index = f"{label_name}_idx_{i:03d}"  # 인덱스를 포함한 라벨 이름 생성

            label_img.save(os.path.join(path_output_label_subtype, f'{label_name_with_index}.png'))  # 라벨 이미지 저장
            label_binary_img.save(os.path.join(path_output_label_binary, f'{label_name_with_index}.png'))  # 라벨 이진화 이미지 저장
            label_binary_img_exclude_Chronic.save(os.path.join(path_output_label_binary_exclude_Chronic, f'{label_name_with_index}.png'))  # 라벨 이진화 이미지 저장 (exclude Chronic)
            label_3class_img.save(os.path.join(path_output_label_3class, f'{label_name_with_index}.png'))  # 라벨 3 클래스 이미지 저장

    # if have_label:
    #     # Print label pixel value range
    #     label_min = label_np.min()
    #     label_max = label_np.max()
    #     print(f'Label pixel value range: {label_min} to {label_max}')

if __name__ == '__main__':
    # path_root_img = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
    # path_root_label = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(unitl_4th)'
    # path_output_img = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\image_png'
    # path_output_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\label_png_subtype'
    # path_output_label_binary = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\label_png_binary'

    # path_root_img = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
    # path_root_label = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_5th)'
    # path_output_img = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\image_png'
    # path_output_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\label_png_subtype'
    # path_output_label_binary = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\label_png_binary'
    # path_output_label_binary_exclude_Chronic = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\label_png_binary_exclude_Chronic'
    # path_output_label_3class = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\1_until_5th\label_png_3class(hemo,chronic)'

    # path_root_img = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
    # path_root_label = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_6th)'
    # path_output_img = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\image_png'
    # path_output_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\label_png_subtype'
    # path_output_label_binary = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\label_png_binary'
    # path_output_label_binary_exclude_Chronic = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\label_png_binary_exclude_Chronic'
    # path_output_label_3class = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\3_until_6th\label_png_3class(hemo,chronic)'

    path_root_img = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
    path_root_label = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_7th)'
    path_output_img = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\image_png'
    path_output_label_subtype = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\label_png_subtype'
    path_output_label_binary = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\label_png_binary'
    path_output_label_binary_exclude_Chronic = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\label_png_binary_exclude_Chronic'
    path_output_label_3class = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\5_until_7th\label_png_3class(hemo,chronic)'


    list_label = os.listdir(path_root_label)

    success_count = 0
    failed_list = []

    for index, cur_image in enumerate(list_label):
        print(f'Processing index: {index} / {len(list_label)-1}', end='\r')

        path_img = os.path.join(path_root_img, cur_image.replace('-label.nrrd', '.nii.gz'))
        path_label = os.path.join(path_root_label, cur_image)

        save_2d_slices(path_img, path_label, path_output_img, path_output_label_subtype,
                       path_output_label_binary, path_output_label_binary_exclude_Chronic,
                       path_output_label_3class, have_label=True)



    # path_root_img = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
    # path_root_label = r''
    # path_output_img = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\0_ALL\image_png'
    # path_output_img_split = r'D:\00000000_Data\20230403_HeuronAnnotation\2_png\0_ALL\image_png(split_by_SeriesID)'
    # path_output_label_subtype = r''
    # path_output_label_binary = r''
    #
    # list_image = os.listdir(path_root_img)
    #
    # success_count = 0
    # failed_list = []
    #
    # for cur_image in list_image:
    #     path_img = os.path.join(path_root_img, cur_image)
    #     path_label = ''
    #
    #     save_2d_slices(path_img, path_label, path_output_img, path_output_label_subtype, path_output_label_binary, have_label=False)
    #
    # # Series ID 기준으로, Split한 폴더에 이미지 저장
    # split_by_seriesID(path_output_img, path_output_img_split)


# import os
# import glob
# import shutil
#
# # Source and destination directories
# src_dir = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_merged_dataset_until_6th"
# dst_dir = r"D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_6th)"
#
# # Get all .nrrd files in the source directory
# nrrd_files = glob.glob(os.path.join(src_dir, "*.nrrd"))
#
# # Move each file to the destination directory
# for file in nrrd_files:
#     print(f"Moving file {file} to {dst_dir}")
#     shutil.copy(file, dst_dir)
