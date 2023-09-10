import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
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
import pandas as pd
from datetime import datetime
import torchio as tio
import random
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, map_coordinates
from random import randint
from skimage import morphology
from PIL import ImageFilter, ImageEnhance
import cv2
from scipy.ndimage import center_of_mass


# Util
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


def calculate_and_log_metrics(output, target, loss, writer, step, epoch=None, batch_idx=None, loader=None, mode="train", do_visualize=False, threshold_output=0.5, num_epochs=None, data=None, sigmoid=True):
    target_np = target.detach().cpu().numpy().astype(np.uint8)

    if sigmoid:
        output_np = (torch.sigmoid(output).detach().cpu().numpy() > threshold_output).astype(np.uint8)
    else:
        output_np = (output.detach().cpu().numpy() > threshold_output).astype(np.uint8)


    tp, fp, tn, fn = get_confusion_matrix_elements(target_np, output_np)

    epsilon = 1e-8
    dice_score = (2 * tp) / ((2 * tp) + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)

    ######################### BF Score #####################
    # Initialize BF Score for batch-level averaging
    batch_bf_score = 0

    # Loop through each sample in the batch
    for i in range(target_np.shape[0]):
        # Single sample extraction
        target_single = target_np[i, 0]
        output_single = output_np[i, 0]

        # Extract boundary pixels
        target_boundary = morphology.dilation(target_single, morphology.disk(1)) - target_single
        output_boundary = morphology.dilation(output_single, morphology.disk(1)) - output_single

        # Calculate True Positive, False Positive, False Negative
        tp_boundary = np.sum(target_boundary * output_boundary)
        fp_boundary = np.sum((1 - target_boundary) * output_boundary)
        fn_boundary = np.sum(target_boundary * (1 - output_boundary))

        # Calculate Precision and Recall
        precision_boundary = tp_boundary / (tp_boundary + fp_boundary + 1e-8)
        recall_boundary = tp_boundary / (tp_boundary + fn_boundary + 1e-8)

        # Calculate BF Score
        bf_score = 2 * (precision_boundary * recall_boundary) / (precision_boundary + recall_boundary + 1e-8)

        # Accumulate BF Score for batch-level averaging
        batch_bf_score += bf_score

    # Average BF Score across the batch
    batch_bf_score /= target_np.shape[0]
    ########################################################

    if mode != 'test':
        writer.add_scalar(f'{mode}_Loss', loss.item(), step)
        writer.add_scalar(f'{mode}_Dice_Score', dice_score, step)
        writer.add_scalar(f'{mode}_IoU', iou, step)
        writer.add_scalar(f'{mode}_Sensitivity', sensitivity, step)
        writer.add_scalar(f'{mode}_Specificity', specificity, step)
        writer.add_scalar(f'{mode}_Accuracy', accuracy, step)
        writer.add_scalar(f'{mode}_BF_Score', batch_bf_score, step)

    log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if mode == 'test':
        print(f"[{mode} || {log_time}] Loss: {loss.item():.4f}, Dice Score: {dice_score:.4f}, IoU: {iou:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}\n")
    else:
        if batch_idx % 10 == 0:
            print(f"[{mode} || {log_time}] Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(loader)}], Total Steps [{step}]")
            print(f"Loss: {loss.item():.4f}, Dice Score: {dice_score:.4f}, IoU: {iou:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}\n")

    if mode != 'test':
        if batch_idx % 20 == 0 and do_visualize:
            images_np = data.detach().cpu().numpy()
            labels_np = target.detach().cpu().numpy()
            outputs_np = torch.sigmoid(output).detach().cpu().numpy()
            visualize_result_with_thresholding(images_np, labels_np, outputs_np)

    # 메트릭을 딕셔너리로 묶어 반환
    metrics = {
        'loss': loss.item(),
        'dice_score': dice_score,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'bf_score': batch_bf_score  # 추가
    }

    return metrics


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, sigmoid=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, y_pred, y_true):
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)

        # Flattening the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (y_pred * y_true).sum()

        dice_coeff = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        return 1.0 - dice_coeff

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, epsilon=1e-7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon  # to avoid division by zero

    def forward(self, y_pred, y_true):
        # Applying sigmoid activation to prediction
        y_pred = torch.sigmoid(y_pred)

        # Flattening the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        tversky_index = (TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon)

        return 1.0 - tversky_index

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def add_noise(image):
    noise = np.random.normal(0, 0.05, image.shape).astype(image.dtype)
    image = (image.astype(np.float64) + noise).clip(0., 255.).astype(image.dtype)
    return image

def transform_sample(sample, transformations):
    image, mask = sample

    # Random Rotation
    if 'RandomRotation' in transformations:
        angle = np.random.uniform(*transformations['RandomRotation'])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

    # Random Horizontal Flip
    if 'RandomHorizontalFlip' in transformations and np.random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random Brightness & Contrast
    if 'RandomBrightnessContrast' in transformations and np.random.random() > 0.5:
        brightness_factor = np.random.uniform(0.7, 1.3)
        contrast_factor = np.random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

    # Random Crop
    if 'RandomCrop' in transformations:
        crop_size = transformations['RandomCrop']
        img_width, img_height = image.size
        crop_width, crop_height = crop_size

        # Calculate the bounding box of the object in the mask
        mask_np = np.array(mask)
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)

        # 예외처리
        if not np.all(rows == False) and np.all(cols == False):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Make sure the crop includes the object
            left = randint(0, min(cmin, img_width - crop_width))
            top = randint(0, min(rmin, img_height - crop_height))
            right = left + crop_width
            bottom = top + crop_height

            image = image.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))

    # Random Rescale
    if 'RandomRescale' in transformations:
        scale_factor = np.random.uniform(*transformations['RandomRescale'])
        target_size = (int(image.height * scale_factor), int(image.width * scale_factor))
        image = TF.resize(image, target_size)
        mask = TF.resize(mask, target_size, interpolation=TF.InterpolationMode.NEAREST)

    # Resize back to 128x128
    image = TF.resize(image, (128, 128))
    # mask = TF.resize(mask, (128, 128))
    mask = TF.resize(mask, (128, 128), interpolation=TF.InterpolationMode.NEAREST)

    image_np = np.array(image)
    mask_np = np.array(mask)

    # Random Elastic Transform
    if 'RandomElasticTransform' in transformations and np.random.random() > 0.5:
        alpha = np.random.uniform(*transformations['RandomElasticTransform']['alpha'])
        sigma = np.random.uniform(*transformations['RandomElasticTransform']['sigma'])
        image_np = elastic_transform(image_np, alpha, sigma)
        mask_np = elastic_transform(mask_np, alpha, sigma)

        # Threshold the mask to keep it binary
        mask_np = (mask_np > 128).astype(np.uint8) * 255

    # Random Noise
    if 'RandomNoise' in transformations and np.random.random() > 0.5:
        image_np = add_noise(image_np)

    image_pil = Image.fromarray(image_np.astype(np.uint8))
    mask_pil = Image.fromarray(mask_np.astype(np.uint8))

    # PIL Image를 텐서로 변환
    image_tensor = TF.to_tensor(image_pil)
    mask_tensor = TF.to_tensor(mask_pil)

    return image_tensor, mask_tensor


# Custom Dataset
class BenignDatasetCSV(Dataset):
    def __init__(self, df, data_mode='ori', do_transform=False, num_randAug=3, num_augmentations=3, do_PP=True):
        self.df = df
        self.do_transform = do_transform
        self.data_mode = data_mode
        self.num_randAug = num_randAug
        self.num_augmentations = num_augmentations
        self.do_PP = do_PP

    def __len__(self):
        if self.do_transform:
            return len(self.df) * (self.num_augmentations + 1)
        else:
            return len(self.df)
    def roi_crop(self, image, mask, method, original_size):
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        def crop_and_restore(img, mask, x, y, w, h, original_size):
            cropped_img = Image.fromarray(img[y:y+h, x:x+w])
            cropped_mask = mask.crop((x, y, x+w, y+h))

            # Restore original size
            restored_img = cropped_img.resize(original_size, Image.ANTIALIAS)
            restored_mask = cropped_mask.resize(original_size, Image.NEAREST)

            return restored_img, restored_mask

        if method == "adaptive_thresholding":
            thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif method == "k_means_clustering":
            flat_gray = gray_image.reshape((-1, 1))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, _ = cv2.kmeans(np.float32(flat_gray), 2, None, criteria, 10, flags)
            labels = labels.reshape(gray_image.shape)
            largest_cluster = np.argmax([np.sum(labels == i) for i in range(2)])
            thresh = np.uint8(labels == largest_cluster) * 255
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif method == "connected_components":
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            thresh = np.uint8(labels == largest_label) * 255
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif method == "simple":
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif method == "complex":
            mask_np = np.array(mask)
            center = center_of_mass(mask_np)
            y_center, x_center = map(int, center)
            x1, x2 = max(x_center - 64, 0), min(x_center + 64, image.size[0])
            y1, y2 = max(y_center - 64, 0), min(y_center + 64, image.size[1])
            return crop_and_restore(gray_image, mask, x1, y1, x2-x1, y2-y1, original_size)
        else:
            raise ValueError("Unknown method: {}".format(method))

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return crop_and_restore(gray_image, mask, x, y, w, h, original_size)
        else:
            return image, mask  # 원본 이미지와 마스크 반환
    def preprocess_image(self, image, mask, do_visualize=False):
        # 원본 이미지와 마스크의 복사본 저장
        original_image = image.copy()
        original_mask = mask.copy()

        # 1. Image Enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)  # Enhance contrast

        # 2. Noise Reduction
        image = image.filter(ImageFilter.MedianFilter(3))

        # 3. ROI Crop
        do_ROI_Crop = False
        if do_ROI_Crop:
            method = "adaptive_thresholding"  # "k_means_clustering", "connected_components", "simple", "complex"
            image, mask = self.roi_crop(image, mask, method, original_size=(128, 128))

        # # 4. Edge Detection
        # image_np = np.array(image)
        # sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
        # edge_detected = np.uint8(np.sqrt(sobel_x ** 2 + sobel_y ** 2))
        # image = Image.fromarray(edge_detected)
        #
        # # 5. Normalization
        # image_np = np.array(image)
        # min_val, max_val = np.min(image_np), np.max(image_np)
        # image_np = 255 * (image_np - min_val) / (max_val - min_val)
        # image = Image.fromarray(np.uint8(image_np))


        if do_visualize:
            # 2x2 격자에 이미지 배치
            plt.subplot(2, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.axis('off')  # x축과 y축 레이블 비활성화
            plt.subplot(2, 2, 2)
            plt.imshow(image, cmap='gray')
            plt.axis('off')  # x축과 y축 레이블 비활성화
            plt.subplot(2, 2, 3)
            plt.imshow(original_mask, cmap='gray')
            plt.axis('off')  # x축과 y축 레이블 비활성화
            plt.subplot(2, 2, 4)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')  # x축과 y축 레이블 비활성화

            plt.show()


        return image, mask

    def __getitem__(self, index):
        row_idx = index // (self.num_augmentations + 1)
        row = self.df.iloc[row_idx]
        if self.data_mode == 'ori':
            img_path = row['Original_img_path']
        else:
            img_path = row['Fuzzy_img_path']
        mask_path = row['Mask_path']

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        ############ 전처리 추가 ############
        if self.do_PP:
            image, mask = self.preprocess_image(image, mask, do_visualize=False)

        if self.do_transform and index % (self.num_augmentations + 1) != 0 and self.num_randAug!=0:
            transformations = {
                'RandomRotation': (-10, 10),
                'RandomHorizontalFlip': None,
                'RandomBrightnessContrast': None,
                'RandomCrop': (100, 100),  # You can set this size according to your needs
                'RandomRescale': (0.8, 1.2),  # Scaling factor range
                'RandomElasticTransform': {'alpha': (0.1, 0.5), 'sigma': (3, 5)},
                'RandomNoise': None
            }

            def rand_augment(transformations, num_selected=3):
                # 딕셔너리의 키를 리스트로 추출
                transform_list = list(transformations.keys())
                # 랜덤하게 선택된 항목의 키 리스트
                selected_keys = random.sample(transform_list, num_selected)
                # 선택된 키를 사용하여 딕셔너리에서 값을 가져옴
                selected_transformations = {key: transformations[key] for key in selected_keys}

                return selected_transformations
            transformations = rand_augment(transformations, num_selected=self.num_randAug)

            image, mask = transform_sample((image, mask), transformations)
        else:
            image = TF.to_tensor(TF.resize(image, (128, 128)))
            mask = TF.to_tensor(TF.resize(mask, (128, 128), interpolation=TF.InterpolationMode.NEAREST))

        return image, mask