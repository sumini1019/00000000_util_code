from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import os
from monai.networks.nets import UNet, UNETR, SwinUNETR
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
# from monai.losses import DiceLoss
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
from utils import *
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import time
import psutil
import torch.onnx
import onnxruntime as ort

# torch.jit.experimental_quantized_execution(backend='cuda')


current_time = datetime.now().strftime("%y%m%d-%H%M%S")
# CUDA 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_mode = 'fuzzy' #'ori' #'fuzzy'
loss_mode = 'Combine' #'BCE' #'Dice' #'Tversky' #'Combine'
model_mode = 'DeepLab_Res101' #'SwinUNETR' #'UNETR' #'DeepLab_Res101' #'UNet'

# Path
if data_mode == 'ori':
    path_src_image = r"C:\Users\user\Downloads\Benign_Renamed\Original_Benign"
else:
    img_folder = r"C:\Users\user\Downloads\Benign_Renamed\Fuzzy_Benign"
path_src_mask = r"C:\Users\user\Downloads\Benign_Renamed\Ground_Truth_Benign_Converted_255"   # 0, 1
path_model_load = r'C:\Users\user\Downloads\test\test_best.pth'

# csv 파일 읽기
data_df = pd.read_csv(r'C:\Users\user\Downloads\Benign_Renamed\20230906_Benign_DataSplit.csv')
# Train, Val, Test로 데이터 분할
train_df = data_df[data_df['Set'] == 'Train']
val_df = data_df[data_df['Set'] == 'Val']
test_df = data_df[data_df['Set'] == 'Test']

# Hyperparameters
mode = 'train'
batch_size = 32 #64
num_epochs = 50
learning_rate = 0.001
do_transform = True
do_visualize = False
do_visualize_Test = True
do_PreProcessing = False
do_PostProcessing = False
do_Check_InferenceTime = False
gpu_check = False

# 최적화
mode_onnx = False
do_Quantization = False
do_Pruning = True
do_HalfPrecision = True

threshold_output = 0.5  # output 이진화 threshold
num_randAug = 3
num_Multi_Aug = 30       # Augmentation 몇배로 늘릴건지
unet_residual = 2 #2    # UNet 내 Residual 수 (0 : 기본 UNet)


name_project = f'7_Final_{current_time}_{data_mode}_{loss_mode}_{model_mode}_PP-{do_PreProcessing}-11000_trans-Rand{num_randAug}_ExcludeBright_MultiAug{num_Multi_Aug}___lr{learning_rate}_th{threshold_output}'
# name_project = f'1_Vanila_{current_time}_Fuzzy_BCE'

# Path #2
path_model_save = os.path.join(r'D:\00000000 Code\00000000_util_code\20230906_output_model_SAM_MED', name_project)
os.makedirs(path_model_save, exist_ok=True)

# Tensorboard
tensorboard_dir = os.path.join(os.getcwd(), 'tensorboard', name_project)
os.makedirs(tensorboard_dir, exist_ok=True)  # 재구성 디렉토리를 생성합니다.
writer = SummaryWriter(log_dir=tensorboard_dir)

# DataSet / Loader 생성
train_dataset = BenignDatasetCSV(train_df, data_mode=data_mode, do_transform=do_transform, num_randAug=num_randAug, num_augmentations=num_Multi_Aug, do_PP=do_PreProcessing)
val_dataset = BenignDatasetCSV(val_df, data_mode=data_mode, do_transform=False, num_randAug=num_randAug, num_augmentations=0, do_PP=do_PreProcessing)
test_dataset = BenignDatasetCSV(test_df, data_mode=data_mode, do_transform=False, num_randAug=num_randAug, num_augmentations=0, do_PP=do_PreProcessing)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



if do_visualize:
    # Load one batch from the DataLoader
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Convert to numpy for visualization
    images = images.numpy()
    labels = labels.numpy()

    # Visualize the batch
    visualize_batch(images, labels)

if loss_mode == 'BCE':
    loss_criterion = nn.BCEWithLogitsLoss()
elif loss_mode == 'Dice':
    loss_criterion = DiceLoss(sigmoid=True)
elif loss_mode == 'Tversky':
    loss_criterion = TverskyLoss(alpha=0.7, beta=0.3)
elif loss_mode == 'Combine':
    weight_bce = 0.6
    weight_dice = 0.4
    loss_criterion_bce = nn.BCEWithLogitsLoss()
    loss_criterion_dice = DiceLoss(sigmoid=True)

train_steps = 0
val_steps = 0
test_steps = 0


if model_mode == 'UNet':
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
elif model_mode == 'DeepLab_Res101':
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
elif model_mode == 'UNETR':
    model = UNETR(spatial_dims=2, in_channels=1, out_channels=1, img_size=(128, 128))
elif model_mode == 'SwinUNETR':
    model = SwinUNETR(spatial_dims=2, in_channels=1, out_channels=1, img_size=(128, 128))

if mode != 'test':
    model.to(device)

    # Adam + StepLR
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # SGD + CosineAnnealingLR
    # optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    for epoch in range(num_epochs):
        ############## Train ###################
        model.train()

        # Initialize epoch-level metrics for training
        epoch_train_loss = 0
        epoch_train_dice_score = 0
        epoch_train_iou = 0
        epoch_train_sensitivity = 0
        epoch_train_specificity = 0
        epoch_train_accuracy = 0
        epoch_train_bf_score = 0  # 추가

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if model_mode == 'DeepLab_Res101':
                output = model(data)['out']
            else:
                output = model(data)

            if loss_mode == 'Combine':
                loss_bce = loss_criterion_bce(output, target)
                loss_dice = loss_criterion_dice(output, target)

                loss = (weight_bce * loss_bce) + (weight_dice * loss_dice)
            else:
                loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()

            # 함수 호출 결과를 metrics 변수에 저장
            metrics = calculate_and_log_metrics(output, target, loss, writer, train_steps, epoch, batch_idx, train_loader, 'train', do_visualize, threshold_output=threshold_output, num_epochs=num_epochs, data=data)

            # Update epoch-level metrics
            epoch_train_loss += metrics['loss']
            epoch_train_dice_score += metrics['dice_score']
            epoch_train_iou += metrics['iou']
            epoch_train_sensitivity += metrics['sensitivity']
            epoch_train_specificity += metrics['specificity']
            epoch_train_accuracy += metrics['accuracy']
            epoch_train_bf_score += metrics['bf_score']  # 추가

            train_steps += 1

        # Calculate average epoch-level metrics for training
        num_batches_train = len(train_loader)
        epoch_train_loss /= num_batches_train
        epoch_train_dice_score /= num_batches_train
        epoch_train_iou /= num_batches_train
        epoch_train_sensitivity /= num_batches_train
        epoch_train_specificity /= num_batches_train
        epoch_train_accuracy /= num_batches_train
        epoch_train_bf_score /= num_batches_train  # 추가

        # Log average epoch-level metrics to TensorBoard
        writer.add_scalar('epoch_train_Loss', epoch_train_loss, epoch)
        writer.add_scalar('epoch_train_Dice_Score', epoch_train_dice_score, epoch)
        writer.add_scalar('epoch_train_IoU', epoch_train_iou, epoch)
        writer.add_scalar('epoch_train_Sensitivity', epoch_train_sensitivity, epoch)
        writer.add_scalar('epoch_train_Specificity', epoch_train_specificity, epoch)
        writer.add_scalar('epoch_train_Accuracy', epoch_train_accuracy, epoch)
        writer.add_scalar('epoch_train_BF_Score', epoch_train_bf_score, epoch)  # 추가
        writer.add_scalar('epoch_learning_rate', optimizer.param_groups[0]['lr'], epoch)


        # 모델 저장
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(path_model_save, fr'ep{epoch}_Dice{epoch_train_dice_score}_IoU{epoch_train_iou}___{name_project}'))




        ############# Validation ###################
        model.eval()

        # Initialize epoch-level metrics for validation
        epoch_val_loss = 0
        epoch_val_dice_score = 0
        epoch_val_iou = 0
        epoch_val_sensitivity = 0
        epoch_val_specificity = 0
        epoch_val_accuracy = 0
        epoch_val_bf_score = 0  # 추가


        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                if model_mode == 'DeepLab_Res101':
                    output = model(data)['out']
                else:
                    output = model(data)

                if loss_mode == 'Combine':
                    loss_bce = loss_criterion_bce(output, target)
                    loss_dice = loss_criterion_dice(output, target)

                    loss = (weight_bce * loss_bce) + (weight_dice * loss_dice)
                else:
                    loss = loss_criterion(output, target)
                metrics = calculate_and_log_metrics(output, target, loss, writer, val_steps, epoch, batch_idx, val_loader, 'val', do_visualize, threshold_output=threshold_output, num_epochs=num_epochs, data=data)

                # Update epoch-level metrics
                epoch_val_loss += metrics['loss']
                epoch_val_dice_score += metrics['dice_score']
                epoch_val_iou += metrics['iou']
                epoch_val_sensitivity += metrics['sensitivity']
                epoch_val_specificity += metrics['specificity']
                epoch_val_accuracy += metrics['accuracy']
                epoch_val_bf_score += metrics['bf_score']  # 추가

                val_steps += 1

        # Calculate average epoch-level metrics for validation
        num_batches_val = len(val_loader)
        epoch_val_loss /= num_batches_val
        epoch_val_dice_score /= num_batches_val
        epoch_val_iou /= num_batches_val
        epoch_val_sensitivity /= num_batches_val
        epoch_val_specificity /= num_batches_val
        epoch_val_accuracy /= num_batches_val
        epoch_val_bf_score /= num_batches_val  # 추가

        # Log average epoch-level metrics to TensorBoard
        writer.add_scalar('epoch_val_Loss', epoch_val_loss, epoch)
        writer.add_scalar('epoch_val_Dice_Score', epoch_val_dice_score, epoch)
        writer.add_scalar('epoch_val_IoU', epoch_val_iou, epoch)
        writer.add_scalar('epoch_val_Sensitivity', epoch_val_sensitivity, epoch)
        writer.add_scalar('epoch_val_Specificity', epoch_val_specificity, epoch)
        writer.add_scalar('epoch_val_Accuracy', epoch_val_accuracy, epoch)
        writer.add_scalar('epoch_val_BF_Score', epoch_val_bf_score, epoch)  # 추가

        # 스케쥴러 추가
        scheduler.step()

        # 모델을 다시 training 모드로 전환 (다음 epoch을 위해)
        model.train()


    #################### Test ###########################
    model.eval()

    # Initialize avg-level metrics for test
    avg_test_dice_score = 0
    avg_test_iou = 0
    avg_test_sensitivity = 0
    avg_test_specificity = 0
    avg_test_accuracy = 0
    avg_test_bf_score = 0  # 추가

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if model_mode == 'DeepLab_Res101':
                output = model(data)['out']
            else:
                output = model(data)

            ################# Convex Hull ################
            if do_PostProcessing:
                output_np = output.detach().cpu().numpy()
                ret, thresh = cv2.threshold(output_np, 0.5, 1, 0)  # You may need to adjust this threshold
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(c)

                    hull_mask = np.zeros(output_np.shape, dtype=np.uint8)
                    cv2.drawContours(hull_mask, [hull], 0, 1, -1)

                    output_np = output_np * hull_mask
                    output = torch.tensor(output_np).to(device)

            if loss_mode == 'Combine':
                loss_bce = loss_criterion_bce(output, target)
                loss_dice = loss_criterion_dice(output, target)

                loss = (weight_bce * loss_bce) + (weight_dice * loss_dice)
            else:
                loss = loss_criterion(output, target)
            # calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test')
            metrics = calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test',
                                                do_visualize, threshold_output=threshold_output, num_epochs=None, data=data)

            # Update avg-level metrics
            avg_test_dice_score += metrics['dice_score']
            avg_test_iou += metrics['iou']
            avg_test_sensitivity += metrics['sensitivity']
            avg_test_specificity += metrics['specificity']
            avg_test_accuracy += metrics['accuracy']
            avg_test_bf_score += metrics['bf_score']  # 추가

            test_steps += 1  # Test 결과 저장

    # Calculate average avg-level metrics for testidation
    num_batches_test = len(test_loader)
    avg_test_dice_score /= num_batches_test
    avg_test_iou /= num_batches_test
    avg_test_sensitivity /= num_batches_test
    avg_test_specificity /= num_batches_test
    avg_test_accuracy /= num_batches_test
    avg_test_bf_score /= num_batches_test  # 추가

    # Log average avg-level metrics to TensorBoard
    writer.add_scalar('avg_test_Dice_Score', avg_test_dice_score, 0)
    writer.add_scalar('avg_test_IoU', avg_test_iou, 0)
    writer.add_scalar('avg_test_Sensitivity', avg_test_sensitivity, 0)
    writer.add_scalar('avg_test_Specificity', avg_test_specificity, 0)
    writer.add_scalar('avg_test_Accuracy', avg_test_accuracy, 0)
    writer.add_scalar('avg_test_BF_Score', avg_test_bf_score, 0)  # 추가

    writer.close()

    print("Training complete.")

else:
    if gpu_check:
        #####################################################################################################################
        import psutil, humanize, GPUtil
        import subprocess as sp
        import os
        from threading import Thread, Timer
        import sched, time

        global chk_list, func_name, p
        func_name = ''
        chk_list = []


        def get_gpu_usage():
            output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
            COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
            try:
                GPU_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
            except sp.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            gpu_use_values = [int(x.split()[0]) for i, x in enumerate(GPU_use_info)]

            COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
            try:
                memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
            except sp.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
            print(gpu_use_values)
            return gpu_use_values[0], memory_use_values[0]


        def get_gpu_usage1():
            gpu = GPUtil.getGPUs()[0]
            return gpu.load * 100, gpu.memoryUsed / 1024


        def check_resource():
            cpu_util = psutil.cpu_percent(None, False)
            gpu_util, VRAM = get_gpu_usage1()
            RAM = psutil.virtual_memory().used / (1024.0 ** 3)
            chk_list.append((func_name, cpu_util, gpu_util, RAM, VRAM))
            # print((func_name, cpu_util, gpu_util, RAM, VRAM))


        def check_resource_1sec():
            if func_name == 'End':
                return
            Timer(0.1, check_resource_1sec).start()
            check_resource()


        func_name = 'Inference Benign'
        check_resource_1sec()



    if mode_onnx:

        # ONNX 모델 로드
        ort_session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'], device_id=0)

        # 입력 이름 가져오기
        input_name = ort_session.get_inputs()[0].name

        #################### Test ###########################

        # Initialize avg-level metrics for test
        avg_test_dice_score = 0
        avg_test_iou = 0
        avg_test_sensitivity = 0
        avg_test_specificity = 0
        avg_test_accuracy = 0
        avg_test_bf_score = 0  # 추가

        with torch.no_grad():
            if do_Check_InferenceTime:
                # Warm-up
                for _ in range(10):
                    data, target = next(iter(test_loader))
                    data, target = data.cpu().numpy(), target.cpu().numpy()  # Convert to numpy array
                    input_data = {input_name: data}
                    _ = ort_session.run(None, input_data)  # ONNX 모델로 warm-up 수행

                # Initialize for VRAM check
                torch.cuda.synchronize()

                total_time = 0
                num_samples = 20  # Number of samples to average over
                time_dict = {}  # 각 단계별 시간을 저장할 딕셔너리

                for _ in range(num_samples):
                    step_time_dict = {}

                    # 데이터 로딩 시간 측정
                    t0 = time.time()
                    data, target = next(iter(test_loader))
                    t1 = time.time()
                    step_time_dict['Data loading'] = t1 - t0

                    # CPU로 데이터 이동 시간 측정
                    t0 = time.time()
                    data, target = data.cpu().numpy(), target.cpu().numpy()  # Convert to numpy array
                    t1 = time.time()
                    step_time_dict['Data to CPU'] = t1 - t0

                    input_data = {input_name: data}

                    # ONNX 추론 전 GPU 동기화
                    torch.cuda.synchronize()
                    t0 = time.time()

                    # ONNX 모델로 추론 수행
                    results = ort_session.run(None, input_data)

                    # ONNX 추론 후 GPU 동기화
                    torch.cuda.synchronize()
                    t1 = time.time()
                    step_time_dict['ONNX Inference'] = t1 - t0

                    total_time += (t1 - t0)

                    time_dict[f'Sample {_+1}'] = step_time_dict

                average_inference_time = total_time / num_samples

                # Model Size
                model_size = os.path.getsize(path_model_load) / (1024 ** 2)  # Size in MB

                print(f"Average ONNX inference time for {num_samples} data points: {average_inference_time:.4f} seconds")
                print(f"Model size: {model_size:.2f} MB")
                print("Time taken for each step:")
                for sample, step_times in time_dict.items():
                    print(f"{sample}: {step_times}")


                if gpu_check:
                    func_name = 'Inference Benign'
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
                    ax[0].grid()
                    ax[1].grid()

                    ax[0].set_ylim(20, 40)
                    ax[0].set_title('RAM')
                    ax[1].set_ylim(0, 11)
                    ax[1].set_title('VRAM')

                    data_dict = {}
                    for i, data in enumerate(chk_list):
                        if data[0] not in data_dict.keys():
                            data_dict[data[0]] = {}
                            data_dict[data[0]]['idx'] = []
                            data_dict[data[0]]['RAM'] = []
                            data_dict[data[0]]['VRAM'] = []

                        data_dict[data[0]]['idx'].append(i)
                        data_dict[data[0]]['RAM'].append(data[3])
                        data_dict[data[0]]['VRAM'].append(data[4])

                    for k in data_dict.keys():
                        idx = data_dict[k]['idx']
                        RAM = data_dict[k]['RAM']
                        VRAM = data_dict[k]['VRAM']
                        ax[0].plot(idx, RAM, label=k)
                        ax[1].plot(idx, VRAM, label=k)
                    ax[0].legend()
                    ax[1].legend()
                    fig.tight_layout()
                    fig.savefig('Usage_vram_memory.png')

                    min_ram = min([data[3] for data in chk_list])
                    max_ram = max([data[3] for data in chk_list])
                    min_vram = min([data[4] for data in chk_list])
                    max_vram = max([data[4] for data in chk_list])

                    print(f"RAM used: {max_ram - min_ram:.2f} GB")
                    print(f"VRAM used: {max_vram - min_vram:.2f} GB")

                exit()

            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                if model_mode == 'DeepLab_Res101':
                    output = model(data)['out']
                else:
                    output = model(data)


                ################# Convex Hull ################
                if do_PostProcessing:
                    # 후처리 때문에, Sigmoid 적용함
                    output = torch.sigmoid(output)

                    output_np = output.detach().cpu().numpy()
                    output_np = np.squeeze(output_np)  # Remove singleton dimensions if any
                    batch_size, height, width = output_np.shape

                    processed_output = np.zeros((batch_size, height, width))

                    for i in range(batch_size):
                        single_output = output_np[i, :, :]
                        ret, thresh = cv2.threshold(single_output, 0.5, 255, 0)  # Adjust this threshold
                        thresh = thresh.astype(np.uint8)  # Convert to uint8

                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        if len(contours) != 0:
                            c = max(contours, key=cv2.contourArea)
                            hull = cv2.convexHull(c)

                            hull_mask = np.zeros(thresh.shape, dtype=np.uint8)
                            cv2.drawContours(hull_mask, [hull], 0, 1, -1)

                            processed_output[i, :, :] = single_output * (hull_mask > 0)

                    output = torch.tensor(processed_output).unsqueeze(1).to(device)  # Add channel dimension back

                if loss_mode == 'Combine':
                    loss_bce = loss_criterion_bce(output, target)
                    loss_dice = loss_criterion_dice(output, target)

                    loss = (weight_bce * loss_bce) + (weight_dice * loss_dice)
                else:
                    loss = loss_criterion(output, target)

                # 2023.09.08
                # - 후처리 (Convex Hull) 때문에 Sigmoid 이미 적용했으므로
                # - Sigmoid = False 로 주고, 함수 내부에서 Sigmoid 안함
                if do_PostProcessing:
                    metrics = calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test',
                                                        do_visualize, threshold_output=threshold_output, num_epochs=None, data=data,
                                                        sigmoid=False)
                else:
                    metrics = calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test',
                                                        do_visualize, threshold_output=threshold_output, num_epochs=None, data=data,
                                                        sigmoid=True)


                # Update avg-level metrics
                avg_test_dice_score += metrics['dice_score']
                avg_test_iou += metrics['iou']
                avg_test_sensitivity += metrics['sensitivity']
                avg_test_specificity += metrics['specificity']
                avg_test_accuracy += metrics['accuracy']
                avg_test_bf_score += metrics['bf_score']  # 추가

                test_steps += 1  # Test 결과 저장


                ############### Test Visualize #################
                if do_visualize_Test:
                    if batch_idx < 5:  # Limit the number of visualizations
                        # Convert tensors to numpy arrays and squeeze singleton dimensions
                        data_np = data.detach().cpu().numpy().squeeze()
                        target_np = target.detach().cpu().numpy().squeeze()

                        if do_PostProcessing:
                            output_np = output.detach().cpu().numpy().squeeze()
                        else:
                            output_np = torch.sigmoid(output).detach().cpu().numpy().squeeze()


                        # Thresholding the output to get a binary mask
                        output_mask = (output_np > 0.5).astype(np.uint8)

                        num_visualize_Test=5
                        for i in range(0, num_visualize_Test):
                            plt.figure(figsize=(12, 4))

                            plt.subplot(1, 4, 1)
                            plt.title('Input Image')
                            plt.imshow(data_np[i], cmap='gray')
                            plt.axis('off')

                            plt.subplot(1, 4, 2)
                            plt.title('Ground Truth')
                            plt.imshow(target_np[i], cmap='gray')
                            plt.axis('off')

                            plt.subplot(1, 4, 3)
                            plt.title('Output Probability')
                            plt.imshow(output_np[i], cmap='gray')
                            plt.axis('off')

                            plt.subplot(1, 4, 4)
                            plt.title('Output Mask')
                            plt.imshow(output_mask[i], cmap='gray')
                            plt.axis('off')

                            plt.tight_layout()
                            plt.show()

        # Calculate average avg-level metrics for testidation
        num_batches_test = len(test_loader)
        avg_test_dice_score /= num_batches_test
        avg_test_iou /= num_batches_test
        avg_test_sensitivity /= num_batches_test
        avg_test_specificity /= num_batches_test
        avg_test_accuracy /= num_batches_test
        avg_test_bf_score /= num_batches_test  # 추가


        # Initialize a dictionary to store average test metrics
        avg_test_metrics = {
            'Dice_Score': avg_test_dice_score,
            'IoU': avg_test_iou,
            'Sensitivity': avg_test_sensitivity,
            'Specificity': avg_test_specificity,
            'Accuracy': avg_test_accuracy,
            'BF_Score': avg_test_bf_score  # 추가
        }

        # Convert the dictionary to a Pandas DataFrame
        df_metrics = pd.DataFrame([avg_test_metrics])

        # Save the DataFrame to a CSV file if needed
        df_metrics.to_csv('test_metrics.csv', index=False)

        # Print the DataFrame
        print("Average Test Metrics:")
        print(df_metrics)

        # Data visualization using matplotlib
        labels = list(avg_test_metrics.keys())
        values = list(avg_test_metrics.values())

        plt.figure(figsize=(12, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('Metric Value')
        plt.title('Average Test Metrics')
        for i, v in enumerate(values):
            plt.text(v, i, f"{v:.4f}", color='black', va='center', ha='left')
        plt.show()

        print(avg_test_metrics)


    else:
        model.load_state_dict(torch.load(path_model_load))
        model.to(device)

        model.eval()

        # Quantization
        if do_Quantization:
            # CPU 에서만 동작하는 문제가 있다고 해서, 임시 변경
            # device = 'cpu'

            # 모델을 quantization-ready 상태로 만듭니다.
            # - 둘 다 실패했음
            # - 'x86' 이 가장 최신이나, 이 역시도 실패
            model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
            # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            # model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

            model = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])

            model = torch.ao.quantization.prepare(model)

            # Calibration
            for batch_idx, (data, target) in enumerate(val_loader):  # val_loader는 검증 데이터셋 로더
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    output = model(data)

            # Quantization 적용
            model = torch.ao.quantization.convert(model)


        # Pruning
        if do_Pruning:
            import torch.nn.utils.prune as prune

            do_AUTO_PRUNING = True
            if do_AUTO_PRUNING:
                def list_layers(model):
                    layer_list = []
                    for name, _ in model.named_modules():
                        layer_list.append(name)
                    return layer_list


                def select_layers_for_pruning(layer_list):
                    layers_to_prune = []
                    # 레이어 이름에 'conv'가 포함되어 있으면 프루닝 대상으로 선정
                    for layer in layer_list:
                        if 'conv' in layer:
                            layers_to_prune.append(layer)
                    return layers_to_prune

                # 모델의 모든 레이어를 나열
                layer_list = list_layers(model)

                # 프루닝을 적용할 레이어를 선별
                layers_to_prune = select_layers_for_pruning(layer_list)

                def apply_and_remove_auto_pruning(model, layers_to_prune, amount=0.2):
                    for layer_name in layers_to_prune:
                        layer = dict(model.named_modules()).get(layer_name, None)
                        if layer is not None:
                            weight = getattr(layer, "weight", None)
                            if weight is None or len(weight.shape) <= 1:
                                continue  # Skip 1D or non-existing weights
                            prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)
                            prune.remove(layer, 'weight')

                # 프루닝 적용 및 불필요한 파라미터 제거
                apply_and_remove_auto_pruning(model, layers_to_prune, amount=0.05)

            else:
                # 선택한 레이어 (예: layer_to_prune)에 대해 가중치의 절대값이 작은 것을 30% 제거
                '''
                L1-norm: L1-norm은 각 가중치의 절대값을 이용하여 중요도를 측정합니다. 이 방법은 가중치 값이 0에 가까울수록 중요도가 낮다고 판단하게 됩니다. 따라서 이상치에 상대적으로 덜 민감하고, 가중치가 0으로 수렴하기 쉽습니다.
                L2-norm: L2-norm은 각 가중치의 제곱을 이용하여 중요도를 측정합니다. 이는 원점에서 벡터까지의 유클리디안 거리를 계산하는 것과 같습니다. 이 방법은 더 큰 가중치 값을 중요하게 여기며, 이상치에 더 민감합니다.
    
                L1-norm은 모델의 해석 가능성을 높이거나, 더 많은 가중치를 정확히 0으로 만들어 모델의 크기를 줄이는 데 유용할 수 있습니다.
                L2-norm은 더 "부드러운" 프루닝을 원할 때 선택되며, 이는 모델의 성능을 덜 저하시킬 수 있습니다.
                '''
                def apply_and_remove_selective_pruning(model, layers_to_prune, amount=0.2, n=2, dim=0):
                    if "initial_conv" in layers_to_prune:
                        initial_conv_layer = model.backbone.conv1
                        prune.ln_structured(initial_conv_layer, name="weight", amount=amount, n=n, dim=dim)

                    if "residual_conv" in layers_to_prune:
                        for layer in model.backbone.layer1:
                            residual_conv_layer = layer.conv1
                            prune.ln_structured(residual_conv_layer, name="weight", amount=amount, n=n, dim=dim)

                    if "aspp_conv" in layers_to_prune:
                        aspp_conv_layer = model.classifier[0].convs[0][0]
                        prune.ln_structured(aspp_conv_layer, name="weight", amount=amount, n=n, dim=dim)

                    if "decoder_conv" in layers_to_prune:
                        decoder_conv_layer = model.decoder.conv1
                        prune.ln_structured(decoder_conv_layer, name="weight", amount=amount, n=n, dim=dim)

                    for name, module in model.named_modules():
                        if name in layers_to_prune:
                            prune.remove(module, 'weight')

                def apply_and_remove_deep_layers_pruning(model, layers_to_prune, amount=0.2):
                    for layer_name in layers_to_prune:
                        layer = getattr(model.backbone, layer_name, None)
                        if layer is None:
                            layer = getattr(model, layer_name, None)
                        if layer is None:
                            continue

                        for block in layer:
                            conv1 = block.conv1
                            prune.l1_unstructured(conv1, name="weight", amount=amount)
                            prune.remove(conv1, 'weight')

                # 프루닝 적용 및 불필요한 파라미터 제거
                apply_and_remove_selective_pruning(model, layers_to_prune=[
                    "initial_conv", "residual_conv", "aspp_conv",
                    "block1_conv", "block2_conv"
                ])
                apply_and_remove_deep_layers_pruning(model, layers_to_prune=[
                    "layer2", "layer3", "layer4",
                    "layer2[-1]", "layer3[-1]", "layer4[-1]"
                ])

            # 프루닝 상태 확인 (옵션)
            for name, module in model.named_modules():
                if 'weight_orig' in dict(module.named_parameters()).keys():
                    print(f"Pruning applied on {name}")

            # 모델 저장 / 로드
            path_model_save = r'C:\Users\user\Downloads\test_pruned.pth'
            torch.save(model.state_dict(), path_model_save)
            model.load_state_dict(torch.load(path_model_save))

        if do_HalfPrecision:
            # Half Precision
            model = model.half()




        #################### Test ###########################
        model.eval()

        # Initialize avg-level metrics for test
        avg_test_dice_score = 0
        avg_test_iou = 0
        avg_test_sensitivity = 0
        avg_test_specificity = 0
        avg_test_accuracy = 0
        avg_test_bf_score = 0  # 추가

        with torch.no_grad():
            if do_Check_InferenceTime:
                # Warm-up
                for _ in range(10):
                    data, target = next(iter(test_loader))
                    if do_HalfPrecision:
                        data, target = data.half(), target.half()
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                # Initialize for VRAM check
                torch.cuda.synchronize()

                total_time = 0
                num_samples = 20  # Number of samples to average over

                time_dict = {}  # 각 단계별 시간을 저장할 딕셔너리

                for _ in range(num_samples):
                    step_time_dict = {}

                    # 데이터 로딩 시간 측정
                    t0 = time.time()
                    data, target = next(iter(test_loader))
                    if do_HalfPrecision:
                        data, target = data.half(), target.half()
                    t1 = time.time()
                    step_time_dict['Data loading'] = t1 - t0

                    # GPU로 데이터 이동 시간 측정
                    t0 = time.time()
                    data, target = data.to(device), target.to(device)
                    t1 = time.time()
                    step_time_dict['Data to GPU'] = t1 - t0

                    # 모델 추론 전 GPU 동기화
                    torch.cuda.synchronize()
                    t0 = time.time()

                    # 모델로 추론 수행
                    output = model(data)

                    # 모델 추론 후 GPU 동기화
                    torch.cuda.synchronize()
                    t1 = time.time()
                    step_time_dict['Model Inference'] = t1 - t0

                    total_time += (t1 - t0)

                    time_dict[f'Sample {_+1}'] = step_time_dict

                average_inference_time = total_time / num_samples

                # Model Size
                import torch


                def get_model_size(model):
                    total_size = 0
                    for param in model.parameters():
                        num_elements = torch.prod(torch.tensor(param.size()))
                        # float16은 2바이트, float32는 4바이트
                        bytes_per_element = param.element_size()
                        total_size += num_elements.item() * bytes_per_element

                    total_size_MB = total_size / (1024 ** 2)
                    return total_size_MB
                # 모델 크기 측정
                model_size = get_model_size(model)

                print(f"Average inference time for {num_samples} data points: {average_inference_time:.4f} seconds")
                print(f"Model size: {model_size:.2f} MB")
                print("Time taken for each step:")
                for sample, step_times in time_dict.items():
                    print(f"{sample}: {step_times}")


                if gpu_check:
                    func_name = 'Inference Benign'
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
                    ax[0].grid()
                    ax[1].grid()

                    ax[0].set_ylim(20, 40)
                    ax[0].set_title('RAM')
                    ax[1].set_ylim(0, 11)
                    ax[1].set_title('VRAM')

                    data_dict = {}
                    for i, data in enumerate(chk_list):
                        if data[0] not in data_dict.keys():
                            data_dict[data[0]] = {}
                            data_dict[data[0]]['idx'] = []
                            data_dict[data[0]]['RAM'] = []
                            data_dict[data[0]]['VRAM'] = []

                        data_dict[data[0]]['idx'].append(i)
                        data_dict[data[0]]['RAM'].append(data[3])
                        data_dict[data[0]]['VRAM'].append(data[4])

                    for k in data_dict.keys():
                        idx = data_dict[k]['idx']
                        RAM = data_dict[k]['RAM']
                        VRAM = data_dict[k]['VRAM']
                        ax[0].plot(idx, RAM, label=k)
                        ax[1].plot(idx, VRAM, label=k)
                    ax[0].legend()
                    ax[1].legend()
                    fig.tight_layout()
                    fig.savefig('Usage_vram_memory.png')

                    min_ram = min([data[3] for data in chk_list])
                    max_ram = max([data[3] for data in chk_list])
                    min_vram = min([data[4] for data in chk_list])
                    max_vram = max([data[4] for data in chk_list])

                    print(f"RAM used: {max_ram - min_ram:.2f} GB")
                    print(f"VRAM used: {max_vram - min_vram:.2f} GB")



                exit()

            for batch_idx, (data, target) in enumerate(test_loader):
                if do_HalfPrecision:
                    data, target = data.half(), target.half()
                data, target = data.to(device), target.to(device)
                if model_mode == 'DeepLab_Res101':
                    output = model(data)['out']
                else:
                    output = model(data)


                ################# Convex Hull ################
                if do_PostProcessing:
                    # 후처리 때문에, Sigmoid 적용함
                    output = torch.sigmoid(output)

                    output_np = output.detach().cpu().numpy()
                    output_np = np.squeeze(output_np)  # Remove singleton dimensions if any
                    batch_size, height, width = output_np.shape

                    processed_output = np.zeros((batch_size, height, width))

                    for i in range(batch_size):
                        single_output = output_np[i, :, :]
                        ret, thresh = cv2.threshold(single_output, 0.5, 255, 0)  # Adjust this threshold
                        thresh = thresh.astype(np.uint8)  # Convert to uint8

                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        if len(contours) != 0:
                            c = max(contours, key=cv2.contourArea)
                            hull = cv2.convexHull(c)

                            hull_mask = np.zeros(thresh.shape, dtype=np.uint8)
                            cv2.drawContours(hull_mask, [hull], 0, 1, -1)

                            processed_output[i, :, :] = single_output * (hull_mask > 0)

                    output = torch.tensor(processed_output).unsqueeze(1).to(device)  # Add channel dimension back

                if loss_mode == 'Combine':
                    loss_bce = loss_criterion_bce(output, target)
                    loss_dice = loss_criterion_dice(output, target)

                    loss = (weight_bce * loss_bce) + (weight_dice * loss_dice)
                else:
                    loss = loss_criterion(output, target)

                # 2023.09.08
                # - 후처리 (Convex Hull) 때문에 Sigmoid 이미 적용했으므로
                # - Sigmoid = False 로 주고, 함수 내부에서 Sigmoid 안함
                if do_PostProcessing:
                    metrics = calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test',
                                                        do_visualize, threshold_output=threshold_output, num_epochs=None, data=data,
                                                        sigmoid=False)
                else:
                    metrics = calculate_and_log_metrics(output, target, loss, writer, test_steps, None, None, None, 'test',
                                                        do_visualize, threshold_output=threshold_output, num_epochs=None, data=data,
                                                        sigmoid=True)


                # Update avg-level metrics
                avg_test_dice_score += metrics['dice_score']
                avg_test_iou += metrics['iou']
                avg_test_sensitivity += metrics['sensitivity']
                avg_test_specificity += metrics['specificity']
                avg_test_accuracy += metrics['accuracy']
                avg_test_bf_score += metrics['bf_score']  # 추가

                test_steps += 1  # Test 결과 저장


                ############### Test Visualize #################
                if do_visualize_Test:
                    if batch_idx < 5:  # Limit the number of visualizations
                        # Convert tensors to numpy arrays and squeeze singleton dimensions
                        data_np = data.detach().cpu().numpy().squeeze()
                        target_np = target.detach().cpu().numpy().squeeze()

                        if do_PostProcessing:
                            output_np = output.detach().cpu().numpy().squeeze()
                        else:
                            output_np = torch.sigmoid(output).detach().cpu().numpy().squeeze()

                        if do_HalfPrecision:
                            data_np = data_np.astype(np.float32)
                            target_np = target_np.astype(np.float32)
                            output_np = output_np.astype(np.float32)

                        # Thresholding the output to get a binary mask
                        output_mask = (output_np > 0.5).astype(np.uint8)

                        num_visualize_Test=5
                        for i in range(0, num_visualize_Test):
                            plt.figure(figsize=(12, 4))

                            plt.subplot(1, 4, 1)
                            plt.title('Input Image')
                            plt.imshow(data_np[i], cmap='gray')
                            plt.axis('off')

                            plt.subplot(1, 4, 2)
                            plt.title('Ground Truth')
                            plt.imshow(target_np[i], cmap='gray')
                            plt.axis('off')

                            plt.subplot(1, 4, 3)
                            plt.title('Output Probability')
                            plt.imshow(output_np[i], cmap='gray')
                            plt.axis('off')

                            plt.subplot(1, 4, 4)
                            plt.title('Output Mask')
                            plt.imshow(output_mask[i], cmap='gray')
                            plt.axis('off')

                            plt.tight_layout()
                            plt.show()

        # Calculate average avg-level metrics for testidation
        num_batches_test = len(test_loader)
        avg_test_dice_score /= num_batches_test
        avg_test_iou /= num_batches_test
        avg_test_sensitivity /= num_batches_test
        avg_test_specificity /= num_batches_test
        avg_test_accuracy /= num_batches_test
        avg_test_bf_score /= num_batches_test  # 추가


        # Initialize a dictionary to store average test metrics
        avg_test_metrics = {
            'Dice_Score': avg_test_dice_score,
            'IoU': avg_test_iou,
            'Sensitivity': avg_test_sensitivity,
            'Specificity': avg_test_specificity,
            'Accuracy': avg_test_accuracy,
            'BF_Score': avg_test_bf_score  # 추가
        }

        # Convert the dictionary to a Pandas DataFrame
        df_metrics = pd.DataFrame([avg_test_metrics])

        # Save the DataFrame to a CSV file if needed
        df_metrics.to_csv('test_metrics.csv', index=False)

        # Print the DataFrame
        print("Average Test Metrics:")
        print(df_metrics)

        # Data visualization using matplotlib
        labels = list(avg_test_metrics.keys())
        values = list(avg_test_metrics.values())

        plt.figure(figsize=(12, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('Metric Value')
        plt.title('Average Test Metrics')
        for i, v in enumerate(values):
            plt.text(v, i, f"{v:.4f}", color='black', va='center', ha='left')
        plt.show()

        print(avg_test_metrics)