import pandas as pd
import os
from sklearn.model_selection import train_test_split
import zipfile

path_src = r'C:\Users\user\Downloads\Benign_Renamed'
final_csv_path = r'C:\Users\user\Downloads\Benign_Renamed\20230906_Benign_DataSplit.csv'

# Update the paths to point to the correct directory
original_img_folder = os.path.join(path_src, 'Original_Benign')
fuzzy_img_folder = os.path.join(path_src, 'Fuzzy_Benign')
mask_folder = os.path.join(path_src, 'Ground_Truth_Benign_Converted_255')

# Create lists to hold the filenames and paths
original_img_names = sorted(os.listdir(original_img_folder))
fuzzy_img_names = sorted(os.listdir(fuzzy_img_folder))
mask_names = sorted(os.listdir(mask_folder))

original_img_paths = [os.path.join(original_img_folder, name) for name in original_img_names]
fuzzy_img_paths = [os.path.join(fuzzy_img_folder, name) for name in fuzzy_img_names]
mask_paths = [os.path.join(mask_folder, name) for name in mask_names]

# Create a DataFrame to hold these details
df = pd.DataFrame({
    'Original_img_name': original_img_names,
    'Original_img_path': original_img_paths,
    'Fuzzy_img_name': fuzzy_img_names,
    'Fuzzy_img_path': fuzzy_img_paths,
    'Mask_name': mask_names,
    'Mask_path': mask_paths
})

# Split the data into training, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2 / 0.9, random_state=42)

# Assign the 'Set' column to specify the type of each image: Train/Val/Test
train_df['Set'] = 'Train'
val_df['Set'] = 'Val'
test_df['Set'] = 'Test'

# Combine all data back into a single DataFrame
final_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

final_df['index'] = final_df['Original_img_name'].str[:3]

# Save the DataFrame to a CSV file
final_df.to_csv(final_csv_path, index=False)