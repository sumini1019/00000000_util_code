### 목적 ###
# - 타겟 폴더의 파일 리스트 로드
# - 원본 폴더에서, 파일명 동일한 파일만, 특정 폴더에 복사


import shutil
import os
import pandas as pd

# 타겟 Json 리스트 csv
path_target_csv = r'H:\20220614_이어명 인턴 전달\20220707_List_Json.csv'

# 타겟 파일 경로 (Json 파일 존재)
path_target = r'C:\Users\user\Downloads\temp_json'

# 원본 폴더 경로 (Label png 파일 존재)
path_ori = r'H:\20220614_이어명 인턴 전달\3. cHS_Segmentation_Dataset\label_binary'

# 복사할 폴더
path_dest = r'C:\Users\user\Downloads\temp_dest'


# # 타겟 파일의 파일 리스트 읽어오기
# list_json = os.listdir(path_target)

# 타겟 Json csv 에서, 파일명 리스트 읽어오기
df = pd.read_csv(path_target_csv)
list_json = list(df['FileName_Json'])

# 카운트
num_suc = 0
num_fail = 0

# 파일 리스트를 반복-순회해서, 파일명 읽어오기
for cur_json in list_json:
    # 현재 Label 파일명으로 변경하기
    cur_fn = cur_json.replace('.json', '_label.png')

    # 현재 Label의 경로 설정
    cur_path_label = os.path.join(path_ori, cur_fn)

    # 현재 Label 파일이, 해당 타겟 폴더에 존재하는지 확인
    if os.path.isfile(cur_path_label):
        # 있다면 -> 복사할 폴더쪽으로 복사하자~

        # 새로운 경로
        cur_path_dest = os.path.join(path_dest, cur_fn)
        # 복사
        shutil.copyfile(cur_path_label, cur_path_dest)

        # 확인용 프린트
        print('Success Copy - {}'.format(cur_path_label))

        # 카운트
        num_suc = num_suc + 1

    else:
        # 확인용 프린트
        print('Error - {}'.format(cur_path_label))

        # 카운트
        num_fail = num_fail + 1

print('성공 - ', num_suc)
print('실패 - ', num_fail)