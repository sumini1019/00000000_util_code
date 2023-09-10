from sklearn.model_selection import KFold
import pandas as pd
import os
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import zipfile

# Initialize variables
path_src = r'C:\Users\user\Downloads\Benign_Renamed'
original_img_folder = os.path.join(path_src, 'Original_Benign')
fuzzy_img_folder = os.path.join(path_src, 'Fuzzy_Benign')
mask_folder = os.path.join(path_src, 'Ground_Truth_Benign_Converted_255')

original_img_names = sorted(os.listdir(original_img_folder))
fuzzy_img_names = sorted(os.listdir(fuzzy_img_folder))
mask_names = sorted(os.listdir(mask_folder))

original_img_paths = [os.path.join(original_img_folder, name) for name in original_img_names]
fuzzy_img_paths = [os.path.join(fuzzy_img_folder, name) for name in fuzzy_img_names]
mask_paths = [os.path.join(mask_folder, name) for name in mask_names]

# Create DataFrame
df = pd.DataFrame({
    'Original_img_name': original_img_names,
    'Original_img_path': original_img_paths,
    'Fuzzy_img_name': fuzzy_img_names,
    'Fuzzy_img_path': fuzzy_img_paths,
    'Mask_name': mask_names,
    'Mask_path': mask_paths
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)

temp_test_df = pd.DataFrame()

for fold, (train_val_idx, test_idx) in enumerate(kf.split(df)):
    test_df = df.iloc[test_idx][:20]
    train_val_df = pd.concat([df.iloc[test_idx][20:], df.iloc[train_val_idx]]).reset_index(drop=True)

    # Split remaining data into training and validation set (7:2 ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=0.22, random_state=42)

    # Assign set type
    train_df['Set'] = 'Train'
    val_df['Set'] = 'Val'
    test_df['Set'] = 'Test'

    # Combine them into a final DataFrame
    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    final_df['Fold'] = fold + 1  # Add fold information

    # Save to CSV
    final_csv_path = os.path.join(path_src, f'20230906_Benign_DataSplit_Fold{fold + 1}.csv')
    final_df.to_csv(final_csv_path, index=False)


    temp_test_df = pd.concat([temp_test_df, test_df]).reset_index(drop=True)

temp_test_df