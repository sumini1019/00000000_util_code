import pandas as pd

# Label 파일
df_label = pd.read_csv(r'Z:\Stroke\DATA\DATA_cELVO_cASPECTS\Final_List_v2.csv')
# Patient ID 조회할 파일
df_ED = pd.read_csv(r'D:\OneDrive\00000000_Code\20221102_cSTROKE\Heuron_EyeDeviation\Result_EyeDeviation.csv')


for i in range(0, len(df_ED)):
    print(i)
    try:
        # 현재 행
        cur_df = df_ED.loc[[i]]

        # 현재 행의, Patient ID 확인
        cur_patientID = cur_df['Patient_ID'].values[0]  # 전체 ID
        cur_direction = cur_df['Patient_ID'].values[0][-1:] # 방향
        cur_ID = cur_df['Patient_ID'].values[0][:-2]    # 방향 제외한 ID

        # Patient ID에 해당하는 Label 확인 (N / R / L)
        cur_label_alphabet = df_label[df_label['HR-ID'] == cur_ID].reset_index(drop=True)
        cur_label_alphabet = cur_label_alphabet['LVO-DR (R / L)'][0]

        # Label 변환 (환자 Direction에 따라, 1/0 변환)
        if cur_label_alphabet == cur_direction:
            cur_label = 1
        else:
            cur_label = 0

        # Label 정보를, df_ED 에 저장
        df_ED._set_value(df_ED['Patient_ID'] == cur_patientID, 'Label_LVO', cur_label)
    except:
        print('Error - ', i)

df_ED.to_csv('temp.csv')