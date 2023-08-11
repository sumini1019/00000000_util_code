import os
import random
import pandas as pd
import nrrd
import numpy as np

# path_label = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230330_merged_dataset_until_4th'
# path_label = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230406_merged_dataset_until_5th'
path_label = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_merged_dataset_until_6th'

# 라벨 파일 리스트
list_label = [f for f in os.listdir(path_label) if f.endswith('.nrrd')]

# Label을 함께 묶어서 shuffle
random.seed(1234)
random.shuffle(list_label)

# 전체 데이터셋의 80%를 train 데이터셋으로 분리
train_size = int(len(list_label) * 0.9)
train_label = list_label[:train_size]

# 전체 데이터셋의 20%를 val+test 데이터셋으로 분리
val_test_label = list_label[train_size:]

# val+test 데이터셋을 10:10 비율로 나눠서 각각 val, test 데이터셋으로 분리
val_size = int(len(val_test_label) * 0.5)
val_label = val_test_label[:val_size]
test_label = val_test_label[val_size:]

# 각 데이터셋의 Label을 함께 묶어서 리스트로 저장
train_data = list(zip(train_label, ['Train']*len(train_label)))
val_data = list(zip(val_label, ['Val']*len(val_label)))
test_data = list(zip(test_label, ['Test']*len(test_label)))

# 데이터셋을 하나의 리스트로 합치고, 이를 데이터프레임으로 변환
data = train_data + val_data + test_data
df = pd.DataFrame(data, columns=['Label', 'Type'])

# Label column의 데이터를 복사하여 Image column 추가
df['Image'] = df['Label'].apply(lambda x: x.replace('-label.nrrd', '.nii.gz'))
df = df[['Image', 'Label', 'Type']]



# selected_classes = ['EDH', 'ICH', 'IVH', 'SAH', 'SDH', 'SDH(Chronic)', 'HemorrhagicContusion']
#
#
# positive_voxels = {class_name: [] for class_name in selected_classes}
#
# for index, row in df.iterrows():
#     # nrrd 파일 읽기
#     label_data, header = nrrd.read(os.path.join(path_label, row['Label']))
#     image_positive_voxels = {class_name: 0 for class_name in selected_classes}
#     for class_name in selected_classes:
#         # Segment name으로부터 label value 얻기
#         class_label_value = [header.get(f"Segment{i}_LabelValue") for i in range(7) if
#                              header.get(f"Segment{i}_Name") == class_name]
#         if class_label_value and len(class_label_value) == 1:
#             class_label_value = int(class_label_value[0])
#             # 양성 voxel 개수 계산
#             class_positive_voxels = np.sum(label_data == class_label_value)
#             image_positive_voxels[class_name] = class_positive_voxels
#             positive_voxels[class_name].append(class_positive_voxels)
#     # DataFrame에 이미지에 대한 클래스 별 양성 voxel 개수 추가 / 라벨 추가
#     for class_name in selected_classes:
#         # Voxel 개수
#         df.loc[index, 'P_Voxel_Num_' + class_name] = image_positive_voxels[class_name]
#
#         # 양/음성 여부
#         if image_positive_voxels[class_name] > 10:
#             df.loc[index, 'Label_' + class_name] = 1
#         else:
#             df.loc[index, 'Label_' + class_name] = 0
#
# df = df.sort_values(by='Image', ascending=True)
#
# # df.to_csv(f'20230331_GT_ICH_Annotation_n{len(df)}.csv', index=False)
#
# path_save = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령'
# df.to_csv(os.path.join(path_save, f'20230406_GT_ICH_Annotation_n{len(df)}_5차수령데이터까지.csv'), index=False)



selected_classes = {1: 'EDH', 2: 'ICH', 3: 'IVH', 4: 'SAH', 5: 'SDH', 6: 'SDH(Chronic)', 7: 'HemorrhagicContusion'}

positive_voxels = {class_name: [] for class_name in selected_classes.values()}

df = df.sort_values(by='Image', ascending=True).reset_index(drop=True)

for index, row in df.iterrows():
    print(f'Processing index: {index} / {len(df)}', end='\r')

    # nrrd 파일 읽기
    label_data, header = nrrd.read(os.path.join(path_label, row['Label']))
    image_positive_voxels = {class_name: 0 for class_name in selected_classes.values()}

    # for class_value, class_name in selected_classes.items():
    #     # Segment name으로부터 label value 얻기
    #     class_label_value = [header.get(f"Segment{i}_LabelValue") for i in range(7) if
    #                          header.get(f"Segment{i}_Name") == class_name]
    #     if class_label_value and len(class_label_value) == 1:
    #         class_label_value = int(class_label_value[0])
    #         # 양성 voxel 개수 계산
    #         class_positive_voxels = np.sum(label_data == class_label_value)
    #         image_positive_voxels[class_name] = class_positive_voxels
    #         positive_voxels[class_name].append(class_positive_voxels)

    for class_value, class_name in selected_classes.items():
        class_positive_voxels = np.sum(label_data == class_value)
        image_positive_voxels[class_name] = class_positive_voxels
        positive_voxels[class_name].append(class_positive_voxels)

    # DataFrame에 이미지에 대한 클래스 별 양성 voxel 개수 추가 / 라벨 추가
    for class_name, class_value in selected_classes.items():
        # Voxel 개수
        df.loc[index, 'P_Voxel_Num_' + class_value] = image_positive_voxels[class_value]

        # 양/음성 여부
        if image_positive_voxels[class_value] > 10:
            df.loc[index, 'Label_' + class_value] = 1
        else:
            df.loc[index, 'Label_' + class_value] = 0

df = df.sort_values(by='Image', ascending=True).reset_index(drop=True)

# df.to_csv(f'20230331_GT_ICH_Annotation_n{len(df)}.csv', index=False)

path_save = r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령'
df.to_csv(os.path.join(path_save, f'20230717_GT_ICH_Annotation_n{len(df)}_6차수령데이터까지.csv'), index=False)
