# 2022.04.22
# - cHS annotion 중, 김두환의 전체 Slice Annotation에서,
# - 특정 아이디들에 해당하는 Annotation만 쭉 뽑아오기 위함

import pandas as pd

# df = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_길병원_Annotation\0_전공의_Annotation_작업\1차작업\김두환(36,000)\Ground_Truth (Only Hemorrhage)_split_0(김두환).csv')
df = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv')
df2 = pd.read_csv(r'Z:\Sumin_Jung\00000000_DATA\1_cHS\20210107_cHS_RSNA_Data\TrainSet_All\rsna_train_sumin.csv')
# df.rename(columns={'ID_Image':'ID_Slice'}, inplace=True)

# df_small_hemo = pd.read_csv(r'Z:\Sumin_Jung\00000000_RESULT\1_cHS\20220422_cHS Small Hemorrhage 논문용 성능 정리\20220425_result_Non-Small_Hemorrhage.csv')
# df_small_hemo = pd.read_csv(r'D:\00000000 Code\20210709_cASPECTS_SW_engine_merge\20220427_result_Small_Hemorrhage(109).csv')
df_small_hemo = pd.read_csv(r'D:\00000000 Code\20210709_cASPECTS_SW_engine_merge\20220428_result_Small_Hemorrhage_2차_100개.csv')
df_small_hemo.rename(columns={'ID':'ID_Slice'}, inplace=True)

# 1차 merge
df_all = df.merge(df2, on='ID_Slice')
# df_all.to_csv('cHS_RSNA_Label_Slice_wise_new_ALL(★★★).csv')

# 2차 merge
df_merge = df_small_hemo.merge(df_all, on='ID_Slice')
df_merge.to_csv(r'Z:\Sumin_Jung\00000000_RESULT\1_cHS\20220422_cHS Small Hemorrhage 논문용 성능 정리\result_Small_Hemorrhage_2차.csv')