import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from monai.networks.nets import UNet
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
from monai.losses import DiceLoss
from monai.metrics import compute_dice
from monai.metrics import compute_meandice, compute_roc_auc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
current_time = datetime.now().strftime("%y%m%d-%H%M%S")

name_project = f'1_Vanila_{current_time}_Orig_BCE'
# name_project = f'1_Vanila_{current_time}_Fuzzy_BCE'

tensorboard_dir = os.path.join(os.getcwd(), './tensorboard', name_project)
os.makedirs(tensorboard_dir, exist_ok=True)  # 재구성 디렉토리를 생성합니다.

writer = SummaryWriter(log_dir=tensorboard_dir)

img_folder = r"C:\Users\user\Downloads\Benign_Renamed\Original_Benign"
# img_folder = r"C:\Users\user\Downloads\Benign_Renamed\Fuzzy_Benign"
mask_folder = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Converted_255"   # 0, 1

# Hyperparameters
num_epochs = 30
learning_rate = 0.001
do_transform = False
do_visualize = False
threshold_output = 0.5  # output 이진화 threshold

# CUDA 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# class UNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNetBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
# class CorrectedUNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(CorrectedUNet, self).__init__()
#
#         # Encoder
#         self.enc1 = UNetBlock(in_channels, 64)
#         self.enc2 = UNetBlock(64, 128)
#         self.enc3 = UNetBlock(128, 256)
#
#         # Decoder
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = UNetBlock(256, 128)
#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = UNetBlock(128, 64)
#
#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
#
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x):
#         # Encoder
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#
#         # Decoder
#         up3 = self.up3(enc3)
#         merge3 = torch.cat([up3, enc2], dim=1)
#         dec3 = self.dec3(merge3)
#
#         up2 = self.up2(dec3)
#         merge2 = torch.cat([up2, enc1], dim=1)
#         dec2 = self.dec2(merge2)
#
#         out = self.final_conv(dec2)
#         return out

# Custom Dataset
class BenignDataset(Dataset):
    def __init__(self, img_folder, mask_folder, transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.img_names = os.listdir(img_folder)
        self.mask_names = os.listdir(mask_folder)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        mask_name = [m for m in self.mask_names if f"{index + 1:03d}" in m][0]

        img_path = os.path.join(self.img_folder, img_name)
        mask_path = os.path.join(self.mask_folder, mask_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def get_confusion_matrix_elements(y_true, y_pred):
    """
    Compute the elements of the confusion matrix
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, fp, tn, fn

# Data transformations
if do_transform:
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
else:
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])



# Initialize datasets and data loaders
benign_dataset = BenignDataset(img_folder, mask_folder, transform=train_transforms)
train_loader = DataLoader(benign_dataset, batch_size=4, shuffle=True)

if do_visualize:
    # Function to visualize a batch of the dataset
    def visualize_batch(images, labels, batch_size=4):
        fig, axes = plt.subplots(2, batch_size, figsize=(15, 5))
        for idx in range(batch_size):
            axes[0, idx].imshow(images[idx][0], cmap='gray')
            axes[0, idx].set_title(f"Image {idx + 1}")
            axes[0, idx].axis('off')

            axes[1, idx].imshow(labels[idx][0], cmap='gray')
            axes[1, idx].set_title(f"Label {idx + 1}")
            axes[1, idx].axis('off')

        plt.show()

    def visualize_result_with_thresholding(images, labels, outputs, batch_size=4, threshold=0.5):
        fig, axes = plt.subplots(4, batch_size, figsize=(15, 10))

        for idx in range(batch_size):
            # Original Images
            axes[0, idx].imshow(images[idx][0], cmap='gray')
            axes[0, idx].set_title(f"Image {idx + 1}")
            axes[0, idx].axis('off')

            # Ground Truth Labels
            axes[1, idx].imshow(labels[idx][0], cmap='gray')
            axes[1, idx].set_title(f"Label {idx + 1}")
            axes[1, idx].axis('off')

            # Model Outputs
            axes[2, idx].imshow(outputs[idx][0], cmap='gray')
            axes[2, idx].set_title(f"Output {idx + 1}")
            axes[2, idx].axis('off')

            # Thresholded Outputs
            thresholded_output = (outputs[idx][0] > threshold).astype(np.uint8)
            axes[3, idx].imshow(thresholded_output, cmap='gray')
            axes[3, idx].set_title(f"Thresholded Output {idx + 1}")
            axes[3, idx].axis('off')

        plt.show()


    # Load one batch from the DataLoader
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Convert to numpy for visualization
    images = images.numpy()
    labels = labels.numpy()

    # Visualize the batch
    visualize_batch(images, labels)

# Initialize the model, loss, and optimizer
# model = CorrectedUNetV2(in_channels=1, out_channels=1)
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
# model = CorrectedUNet(1, 1)

model.to(device)  # 모델을 GPU로 이동


loss_criterion = nn.BCEWithLogitsLoss()
# loss_criterion = DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

dice_loss = DiceLoss(sigmoid=True)

total_steps = 0
# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 데이터와 타겟을 GPU로 이동
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute Loss
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate metrics
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        output_np = (torch.sigmoid(output).detach().cpu().numpy() > threshold_output).astype(np.uint8)

        tp, fp, tn, fn = get_confusion_matrix_elements(target_np, output_np)

        epsilon = 1e-8
        dice_score = (2 * tp) / ((2 * tp) + fp + fn + epsilon)
        iou = tp / (tp + fp + fn + epsilon)
        accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)
        sensitivity = tp / (tp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss', loss.item(), total_steps)
        writer.add_scalar('Dice_Score', dice_score.item(), total_steps)
        writer.add_scalar('IoU', iou, total_steps)
        writer.add_scalar('Sensitivity', sensitivity, total_steps)
        writer.add_scalar('Specificity', specificity, total_steps)
        writer.add_scalar('Accuracy', accuracy, total_steps)

        total_steps += 1  # 총 steps 업데이트

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Total Steps [{total_steps}]")
            print(f"Loss: {loss.item():.4f}, Dice Score: {dice_score.item():.4f}, IoU: {iou:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}")

        # For visualization: Convert the tensors to numpy arrays
        if batch_idx % 20 == 0 and do_visualize:
            images_np = data.detach().cpu().numpy()
            labels_np = target.detach().cpu().numpy()
            outputs_np = torch.sigmoid(output).detach().cpu().numpy()
            # visualize_result(images_np, labels_np, outputs_np)
            # visualize_result_with_overlay(images_np, labels_np, outputs_np)
            if do_visualize:
                visualize_result_with_thresholding(images_np, labels_np, outputs_np)

writer.close()

print("Training complete.")