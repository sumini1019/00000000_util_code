import chardet
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from glob import glob


# 2023.03.15
# - pd.read_csv() 대체
# - 파일 encoding 자동 검사 후, csv 리턴
def read_csv_autodetect_encoding(filename, sample_size=10000):
    with open(filename, 'rb') as f:
        sample = f.read(sample_size)
        result = chardet.detect(sample)

    return pd.read_csv(filename, encoding=result['encoding'])

# 2023.04.20
# - ROC Curve 그리기
# - AUC 산출
def plot_roc_curve(labels, probs, title, draw_ROC=False):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    if draw_ROC:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {title}')
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc


# 2023.04.21
# - Dicom Series로부터, nifti 생성하는 함수
# - ex) make_nifti_from_dicom(path_input, path_output, save=True)
def make_nifti_from_dicom(path_input, path_output, save=False):
    try:
        series_ID = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_input)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_input, series_ID[0])
    except:
        series_file_names = glob(path_input + '/*.dcm')
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image = series_reader.Execute()

    if save:
        # Nifti 형식으로 이미지 저장
        sitk.WriteImage(image, os.path.join(path_output, f"{os.path.basename(path_input)}.nii.gz"))

    return image