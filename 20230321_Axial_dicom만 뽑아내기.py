import os
import pydicom
import shutil

# 데이터 폴더 경로
data_dir = r'Z:\Sumin_Jung\00000000_DATA\6_cSTROKE\20230322_HeuronStroke_v012_태국데이터 테스트셋\태국 NCCT (3 vendors)-20230321T063517Z-001\태국 NCCT (3 vendors)\원본\Siemens\Stroke 2\image'
path_dest_root = r'Z:\Sumin_Jung\00000000_DATA\6_cSTROKE\20230322_HeuronStroke_v012_태국데이터 테스트셋\태국 NCCT (3 vendors)-20230321T063517Z-001\태국 NCCT (3 vendors)\only_Axial'

fn = data_dir.split('\\')[8] + '_' + data_dir.split('\\')[9]

# axial cut 이미지 리스트
axial_list = []

# axial cut 이미지의 z축 좌표값 리스트
z_coords = []

# 폴더 내 모든 dicom 파일에 대해서
for file in os.listdir(data_dir):
    print(f'File Name : {os.path.basename(file)}')
    if file.endswith('.dcm'):
        file_path = os.path.join(data_dir, file)
        ds = pydicom.dcmread(file_path)

        try:
            image_type = ds.ImageType
            if image_type[2] != 'AXIAL':
                continue
        except AttributeError:
            continue

        # axial cut 이미지일 경우
        if ds.ImageType[2] == 'AXIAL':
            axial_list.append(file_path)
            z_coords.append(ds.ImagePositionPatient[2])

# axial cut 이미지의 z축 좌표값을 기준으로 정렬
axial_list = [x for _, x in sorted(zip(z_coords, axial_list))]

print(axial_list)

import os
import shutil

path_temp = r'Z:\Sumin_Jung\00000000_DATA\6_cSTROKE\20230322_HeuronStroke_v012_태국데이터 테스트셋\태국 NCCT (3 vendors)-20230321T063517Z-001\태국 NCCT (3 vendors)\temp'
os.makedirs(path_temp, exist_ok=True)

# 리스트에 저장된 파일 경로를 순회하며 복사
for file_path in axial_list:
    # 파일 이름만 추출
    file_name = os.path.basename(file_path)
    # 파일을 복사하여 path_dest에 저장
    shutil.copy(file_path, os.path.join(path_temp, file_name))



# Axial Cut을 1개 시리즈로 만들어주기
import os
import pydicom

# 파일 리스트
file_list = os.listdir(path_temp)  # 파일 경로 리스트

path_dest_new = os.path.join(path_dest_root, fn)
os.makedirs(path_dest_new, exist_ok=True)

# 새로운 정보
new_patient_id = 'GE_Sample_1'
new_accession_number = '00000001'
new_patient_name = 'GE_Sample_1_PatientName'
new_study_instance_uid = '1.2.345.67891011121314151617181920'
new_series_instance_uid = '1.2.345.67891011121314151617181921'
new_sop_instance_uid_base = '1.2.345.67891011121314151617181922.'
new_study_description = 'GE_Sample_1_StudyDescription'
new_series_description = 'GE_Sample_1_SeriesDescription'

# 모든 파일에 대해서 수정
for i, item in enumerate(file_list):
    file_path = os.path.join(path_temp, item)

    ds = pydicom.dcmread(file_path)
    ds.PatientID = new_patient_id
    ds.AccessionNumber = new_accession_number
    ds.PatientName = new_patient_name
    ds.StudyInstanceUID = new_study_instance_uid
    ds.SeriesInstanceUID = new_series_instance_uid
    ds.SOPInstanceUID = new_sop_instance_uid_base + str(i+1).zfill(4)
    ds.StudyDescription = new_study_description
    ds.SeriesDescription = new_series_description
    ds.save_as(os.path.join(path_dest_new, item))  # 수정한 내용 저장

shutil.rmtree(path_temp)
