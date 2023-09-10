from PIL import Image
import numpy as np
import glob
import os

# Path to the Ground_Truth_Benign folder
ground_truth_folder_path = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign"  # 실제 경로로 변경해 주세요.

# Initialize a variable to check if all images contain only 0s and 1s
all_binary_values = True
non_binary_files = []

# Loop through all image files in the Ground_Truth_Benign folder
for img_path in glob.glob(f"{ground_truth_folder_path}/*.png"):
    # Open the image file
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Check if the image contains only 0s and 1s
    unique_values = np.unique(img_array)
    if not np.all(np.isin(unique_values, [1, 2])):
        all_binary_values = False
        non_binary_files.append(os.path.basename(img_path))

print("All files have only binary values:", all_binary_values)
if not all_binary_values:
    print("Files with non-binary values:", non_binary_files)
