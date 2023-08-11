import os
import pandas as pd


# 전체 이미지 데이터 경로
path_image = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\image_Hemo_nifti (ALL)'
# 1~5차 데이터 label 경로
path_label = r'D:\00000000_Data\20230403_HeuronAnnotation\1_nifti_nrrd\label_nrrd(until_5th)'

# 모든 리스트
list_image = os.listdir(path_image)
list_label = os.listdir(path_label)

# label이 이미 있는건 제거
for cur_label in list_label:
    fn = cur_label.split('-label.nrrd')[0]

    fn_image = fn + '.nii.gz'

    if fn_image in list_image:
        list_image.remove(fn_image)
    else:
        print(f'Label에 해당하는 이미지가 없음 ({fn_image})')

df_not_have_Label = pd.DataFrame(list_image, columns=['ID_not_have_Label'])
df_not_have_Label.to_csv('20230425_Label없는_RSNA_SeriesID리스트_5차데이터셋기준.csv', index=False)