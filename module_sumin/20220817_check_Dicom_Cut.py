# 2022.08.17
# - Dicom 시리즈에서, Axial Cut 외 데이터가 있는지 확인

from glob import glob
import pydicom

def check_series_isAxial(path_dcm):
    # 데이터 경로 리스트
    list_dcm = glob(path_dcm + '/*.dcm')

    # 시리즈가 모두 Axial 인가?
    isAxial = True

    def check_image_position(cur_dcm, next_dcm):
        cur_pos = cur_dcm[('0020', '0032')].value
        next_pos = next_dcm[('0020', '0032')].value

        diff_pos = abs(cur_pos[2] - next_pos[2])

        if (diff_pos >= 0) and (diff_pos <= 8):
            return True
        else:
            return False

    for i in range(0, len(list_dcm)-1):
        cur_dcm = pydicom.read_file(list_dcm[i])
        next_dcm = pydicom.read_file(list_dcm[i+1])

        # Image Position 검사해서, 일정 이상 차이 날 경우 Axial Series 아님
        if not check_image_position(cur_dcm, next_dcm):
            isAxial = False

    return isAxial


isAxial = check_series_isAxial(path_dcm = r'Z:\Stroke\DATA\cASPECTS_Clinical_Evaluation\음성군(뇌질환 X)\G159')

print(isAxial)