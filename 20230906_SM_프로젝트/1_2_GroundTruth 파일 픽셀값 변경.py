from PIL import Image
import numpy as np
import glob
import os


def change_and_save_images(ground_truth_folder_path, dest_folder):
    # Initialize a variable to check if all images contain only 1s and 2s
    all_binary_values = True
    non_binary_files = []

    # Make the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Loop through all image files in the Ground_Truth_Benign folder
    for img_path in glob.glob(f"{ground_truth_folder_path}/*.png"):
        # Open the image file
        img = Image.open(img_path)

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Check if the image contains only 1s and 2s
        unique_values = np.unique(img_array)
        if not np.all(np.isin(unique_values, [1, 2])):
            all_binary_values = False
            non_binary_files.append(os.path.basename(img_path))
            continue  # Skip this file and proceed to the next one

        # Change pixel values: 1 to 0 and 2 to 1
        img_array[img_array == 1] = 0
        img_array[img_array == 2] = 255

        # Save the new image
        new_img = Image.fromarray(img_array.astype('uint8'))
        new_img_path = os.path.join(dest_folder, os.path.basename(img_path))
        new_img.save(new_img_path)

    return all_binary_values, non_binary_files


# Example usage (please change these paths to your actual folder paths)
ground_truth_folder_path = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign"
dest_folder = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Converted_255"

all_binary_values, non_binary_files = change_and_save_images(ground_truth_folder_path, dest_folder)
