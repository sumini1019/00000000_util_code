# 2023.08.23
# 1.
# - 전체 데이터에서 약 10%의 테스트 세트를 추출
# - 테스트 세트는 데이터 품질(높음, 중간, 낮음) 기준으로 1:1:1 비율로 추출하며, 각 비율 내에서 LVO 양성/음성이 균등하게 분포하도록 추출
# 2.
# - 전체 데이터에서 테스트셋 제외한 나머지 데이터로 Train과 Val 데이터셋을 생성
# - 총 3개 Fold의 Train/Val 데이터셋을 구성하는데, 각 3개의 폴드 구성은 랜덤으로 독립적이며 중복될 수 있음
# - Train은 90%, Val은 10%의 비율
# - 각 Train과 Val 데이터셋은 데이터 품질 및 LVO 양성/음성 비율을 비슷하게 유지하도록 샘플링
# - 모든 생성된 세트는 전체 데이터셋의 'index' 열을 기준으로 정렬 후 csv 저장


import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# CSV 파일 읽기
file_path = r"Z:\Stroke\DATA\DATA_cELVO_cASPECTS\20230615_FinalList_v4_데이터퀄리티_포함.csv"
df = pd.read_csv(file_path)

# 퀄리티 별로 데이터 분리
df_high = df[df["Quality"] == "High"]
df_medium = df[df["Quality"] == "Medium"]
df_low = df[df["Quality"] == "Low"]

# 각 퀄리티 그룹별 샘플 수 계산 (1:1:1 비율)
num_samples_per_quality = int(len(df) * 0.10) // 3

# LVO 양성과 음성 케이스를 균등하게 샘플링하는 함수
def sample_evenly(df, num_samples):
    df_pos = df[df["GT-LVO (Y:1 / N:0)"] == 1].sample(num_samples // 2)
    df_neg = df[df["GT-LVO (Y:1 / N:0)"] == 0].sample(num_samples - num_samples // 2)
    return pd.concat([df_pos, df_neg])

# 각 퀄리티 그룹에서 데이터 샘플링
test_set_high = sample_evenly(df_high, num_samples_per_quality)
test_set_medium = sample_evenly(df_medium, num_samples_per_quality)
test_set_low = sample_evenly(df_low, num_samples_per_quality)

# 'HR-ID'에서 'ID' 및 'MC' 열을 추출하는 함수
def extract_columns(df):
    df['ID'] = df['HR-ID']
    df['MC'] = df['HR-ID'].apply(lambda x: '1_AJUMC' if x[3] == 'A' else
                                           '2_GMC' if x[3] == 'G' else
                                           '3_EUMC' if x[3] == 'E' else
                                           '4_CSUH' if x[3] == 'C' else
                                           '5_CNUSH' if x[3] == 'S' else
                                           '6_SCHMC' if x[3] == 'B' else
                                           '7_etc')
    return df

# 테스트 세트 병합
test_set = pd.concat([test_set_high, test_set_medium, test_set_low])
# TestSet 열 추가
test_set = extract_columns(test_set)

# CSV로 저장
test_set_path = r"Z:\Stroke\DATA\DATA_cELVO_cASPECTS\label_csv\20230823_ver2\20230823_test_set.csv"
test_set = test_set.sort_values('index')
test_set.to_csv(test_set_path, index=False)

# 테스트 세트를 제외한 데이터로 Train과 Val 데이터셋 생성
train_val_df = df.loc[~df.index.isin(test_set.index)]
train_val_df['quality_lvo'] = train_val_df['Quality'] + "_" + train_val_df['GT-LVO (Y:1 / N:0)'].astype(str)

# Stratified Shuffle Split을 사용해 3개의 폴드 생성
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.10, random_state=42)

for fold, (train_index, val_index) in enumerate(sss.split(train_val_df, train_val_df['quality_lvo'])):
    # Train/val 정렬 및 추가 열 생성
    train_fold = extract_columns(train_val_df.iloc[train_index].sort_values('index'))
    val_fold = extract_columns(train_val_df.iloc[val_index].sort_values('index'))

    # CSV 파일로 저장
    train_path = fr"Z:\Stroke\DATA\DATA_cELVO_cASPECTS\label_csv\20230823_ver2\20230823_train_fold_{fold+1}.csv"
    val_path = fr"Z:\Stroke\DATA\DATA_cELVO_cASPECTS\label_csv\20230823_ver2\20230823_val_fold_{fold+1}.csv"
    train_fold.drop(columns=['quality_lvo']).to_csv(train_path, index=False)
    val_fold.drop(columns=['quality_lvo']).to_csv(val_path, index=False)

    # 품질 및 LVO 비율 출력
    for dataset, name in [(train_fold, 'Train'), (val_fold, 'Val')]:
        print(f"{name} fold {fold+1} statistics:")
        print("Quality ratio:")
        print(dataset['Quality'].value_counts(normalize=True))
        print("Quality-LVO ratio:")
        for quality in ['High', 'Medium', 'Low']:
            print(f"  {quality}: {dataset[dataset['Quality'] == quality]['GT-LVO (Y:1 / N:0)'].value_counts(normalize=True).to_dict()}")
