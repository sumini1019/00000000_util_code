# 2023.07.17
# - Annotation 파일에, 특정한 파일명들을 일원화해서 수정하기 위한 작업
# - 폴더의 모든 파일명 검사해서, '_1-label' / '_2-label' / '_3-label' 과 같은 파일명을 '-label' 로 변경함

import os
import glob

# 변환할 디렉토리 설정
directory = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\20230120_탑병원 Annotation 결과 수령\20230717_6차 수령 데이터\SDH-A"

# 디렉토리에서 모든 파일을 가져옴
for filename in glob.glob(os.path.join(directory, '*')):
    # 파일 이름에서 '_2-label', '_1-label', '_3-label'를 찾아서 '-label'로 변경
    new_filename = filename.replace('_2-label', '-label').replace('_1-label', '-label').replace('_3-label', '-label')

    # 파일 이름이 바뀌었는지 확인
    if filename != new_filename:
        print(f"Renaming file {filename} to {new_filename}")
        os.rename(filename, new_filename)
