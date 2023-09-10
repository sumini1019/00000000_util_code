from PIL import Image
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Initialize the folder path and check if it exists
source_folder_path = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Converted_255"  # Please replace with your actual source folder path
target_folder_path = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Colormap_Applied_255"  # Please replace with your desired target folder path

# Create the target folder if it doesn't exist
os.makedirs(target_folder_path, exist_ok=True)

# Define the new color map (LUT)
# Black for 1: [0, 0, 0]
# Red for 2: [255, 0, 0]
# Yellow for others: [255, 255, 0]
new_LUT = [255, 255, 0] * 256  # Default is yellow
new_LUT[0:3] = [0, 0, 0]  # Black for 0
new_LUT[3:6] = [255, 0, 0]  # Red for 1

# Loop through all the PNG files in the folder
for img_path in glob.glob(f"{source_folder_path}/*.png"):
    # Open the image
    img = Image.open(img_path).convert("P")

    # Apply the new LUT
    img.putpalette(new_LUT)

    # Save the modified image in the new folder
    file_name = os.path.basename(img_path)
    new_img_path = os.path.join(target_folder_path, file_name)
    img.save(new_img_path)