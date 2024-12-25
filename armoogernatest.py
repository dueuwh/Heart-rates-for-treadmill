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

label_index = "6sec_plot"
order = "5th"

base_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/"
algorithms = os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/")
rppg_bpm_pred_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/results/rppg_toolbx_hr_5th_fft/"

labels_total = natsort.natsorted([name for name in os.listdir(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/") if "label" in name])
labels = natsort.natsorted([name for name in os.listdir(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/") if "label" in name])
labels = natsort.natsorted([name for name in labels if "_3_" in name])

data_total = {}

for algorithm in algorithms:
    gt_list_total = natsort.natsorted([name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "gt" in name])
    pr_list_total = natsort.natsorted([name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "bvp" in name])

    gt_list = []
    pr_list = []

    for i in range(len(labels_total)):
        if "_3_" in labels_total[i]:
            gt_list.append(gt_list_total[i])
            pr_list.append(pr_list_total[i])
    
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
                
                
                
                
                