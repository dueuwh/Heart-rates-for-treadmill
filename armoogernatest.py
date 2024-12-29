from copy import deepcopy
import math
import cv2
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from params import Params
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import natsort
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import sys
import seaborn as sns
from PyEMD import CEEMDAN
from scipy.fft import fft, fftfreq
import importlib.util

def import_function_from_file(file_path, function_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

data_total, algorithms = load_dataset()

path = "D:/home/BCML/IITP/rPPG-Toolbox/evaluation/post_process.py"
function = "_calculate_fft_hr"
cal_hr_fft = import_function_from_file(path, function)
sampling_rate = 30.0
window_seconds = 6
window_size = int(sampling_rate*window_seconds)

save_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/results/rppg_toolbx_hr_5th_fft/"
for algorithm in algorithms:
    os.makedirs(f"{save_path}{algorithm}", exist_ok=True)

for key in data_total.keys():
    for algorithm in algorithms:
        bvp = data_total[key][algorithm]['pred']
        label = data_total[key][algorithm]['label']
        hr_list = []
        for i in range(len(bvp)-window_size):
            hr_list.append(cal_hr_fft(bvp[i:i+window_size]))
        # np.save(f"{save_path}{algorithm}/{key}.npy", np.array(hr_list))
        ax_pred_hr = np.linspace(180, len(hr_list)+180, len(hr_list))
        plt.title(f"{key}, {algorithm}, Prediction length: {len(hr_list)}, prediction ax first point: {ax_pred_hr[0]}, prediction ax last point: {ax_pred_hr[-1]} label length: {len(label)}")
        plt.plot(label, label="label")
        plt.plot(ax_pred_hr, hr_list, label=f"predicted HR of {algorithm}")
        plt.show()
        
        


label_index = "6sec_plot"
order = "5th"

base_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/"
algorithms = os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/")
rppg_bpm_pred_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/results/rppg_toolbx_hr_5th_fft/"

labels = natsort.natsorted([name for name in os.listdir(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/") if "label" in name])

for label in labels:
    temp_label = np.load(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/{label}")
    # plt.plot(temp_label)
    # plt.show()

data_total = {}

for algorithm in algorithms:
    gt_list = natsort.natsorted([name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "gt" in name])
    pr_list = natsort.natsorted([name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "bvp" in name])
    snr_list = natsort.natsorted([name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "SNR" in name])
    
    for i in range(len(gt_list)):
        temp_hr_gt = np.load(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}/{gt_list[i]}")
        temp_bvp_pr = np.load(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}/{pr_list[i]}")
    
        temp_label = np.load(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/{labels[i]}")
        
        temp_key_word_list = labels[i].split('_')
        temp_key = f"{temp_key_word_list[0]}_{temp_key_word_list[1]}_{temp_key_word_list[2]}_{temp_key_word_list[3]}_"
        
        if temp_key not in data_total:
            data_total[temp_key] = {}
        if algorithm not in data_total[temp_key]:
            data_total[temp_key][algorithm] = {}
        
        data_total[temp_key][algorithm]['pred'] = np.load(f"{rppg_bpm_pred_path}{algorithm}/{temp_key}.npy").tolist()
        
        if 'label' not in data_total[temp_key][algorithm]:
            data_total[temp_key][algorithm]['label'] = temp_label.tolist()
        else:
            if len(temp_label.tolist()) > 1:
                data_total[temp_key][algorithm]['label'].extend(temp_label)
            else:
                data_total[temp_key][algorithm]['label'].append(temp_label.tolist()[0])

        temp_SNR = np.load(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}/{snr_list[i]}")

        if 'SNR' not in data_total[temp_key][algorithm]:
            data_total[temp_key][algorithm]['SNR'] = temp_SNR.tolist()
        else:
            if len(temp_SNR.tolist()) > 1:
                data_total[temp_key][algorithm]['SNR'].extend(temp_SNR)
            else:
                data_total[temp_key][algorithm]['SNR'].append(temp_SNR.tolist()[0])


