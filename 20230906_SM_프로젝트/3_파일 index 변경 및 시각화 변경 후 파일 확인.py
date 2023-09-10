from PIL import Image
import os
import matplotlib.pyplot as plt


# Modified function to accurately display images with the same index from different folders
def display_images_with_accurate_index_3_digits(index, folder_paths):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Convert index to a 3-digit string
    index_3_digits = f"{index:03d}"

    for ax, folder_path in zip(axs, folder_paths):
        folder_name = os.path.basename(folder_path)

        # Find the image that contains the 3-digit index in its name
        found = False
        for img_name in os.listdir(folder_path):
            if index_3_digits in img_name:
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path)
                ax.imshow(img, cmap='gray')
                found = True
                break

        if not found:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')

        ax.set_title(folder_name)
        ax.axis('off')

    plt.show()


# Define folder paths
folder_paths = [
    r"C:\Users\user\Downloads\Benign_Renamed\Fuzzy_Benign",
    r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Converted_255",
    r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Colormap_Applied_255",
    r"C:\Users\user\Downloads\Benign_Renamed\Original_Benign"
]

for index in range(1, 20): #201):  # Loop from 1 to 200
    display_images_with_accurate_index_3_digits(index, folder_paths)
    # To pause between each set of images, you can uncomment the next line
    # input("Press Enter to continue...")
