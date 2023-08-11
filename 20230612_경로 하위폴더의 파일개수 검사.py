import os
import glob
import pandas as pd

parent_folder = r"Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_Split_by_SeriesUID"
list_folder = os.listdir(parent_folder)

data = []
for cur_folder in list_folder:
    list_dcm = glob.glob(os.path.join(parent_folder, cur_folder) + '/*.dcm')
    data.append((cur_folder, len(list_dcm)))

df = pd.DataFrame(data, columns=['folder', 'num_dcm'])
df.to_csv("20230612_ICH폴더 별 파일개수.csv", index=False)
