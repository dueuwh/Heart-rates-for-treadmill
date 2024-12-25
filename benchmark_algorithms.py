import os
from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
import natsort

label_index = "3sec"
order = "6th"

base_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/"
algorithms = os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/")

labels = natsort.natsorted([name for name in os.listdir(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/") if "label" in name])

for label in labels:
    temp_label = np.load(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/{label}")
    # plt.plot(temp_label)
    # plt.show()

data_total = {}

for algorithm in algorithms:
    gt_list = [name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "gt" in name]
    pr_list = [name for name in os.listdir(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}") if "pre" in name]
    
    
    for i in range(len(gt_list)):
        temp_hr_gt = np.load(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}/{gt_list[i]}")
        temp_hr_pr = np.load(f"{base_path}results/rppg_toolbox_hr_{order}/{algorithm}/{pr_list[i]}")
    
        temp_label = np.load(f"{base_path}rppg_toolbox/preprocess4hr_{label_index}/datafiles/{labels[i]}")
        
        temp_key = labels[i].split('_')[0]
        if temp_key not in data_total:
            data_total[temp_key] = {}
        if algorithm not in data_total[temp_key]:
            data_total[temp_key][algorithm] = {}
        
        if 'pred' not in data_total[temp_key][algorithm]:
            if len(temp_hr_pr.tolist()) > 1:
                data_total[temp_key][algorithm]['pred'] = [round(sum(temp_hr_pr)/len(temp_hr_pr), 4)]
            else:
                data_total[temp_key][algorithm]['pred'] = temp_hr_pr.tolist()
        else:
            if len(temp_hr_pr.tolist()) > 1:
                data_total[temp_key][algorithm]['pred'].append(round(sum(temp_hr_pr)/len(temp_hr_pr), 4))
            else:
                data_total[temp_key][algorithm]['pred'].append(temp_hr_pr.tolist()[0])
        if 'label' not in data_total[temp_key][algorithm]:
            data_total[temp_key][algorithm]['label'] = temp_label.tolist()
        else:
            if len(temp_label.tolist()) > 1:
                data_total[temp_key][algorithm]['label'].extend(temp_label)
            else:
                data_total[temp_key][algorithm]['label'].append(temp_label.tolist()[0])
        

data_chunk = 90  # 30 * 6 (6seconds)

for key in data_total.keys():
    for algorithm in algorithms:
        plt.title(f"{key} | {algorithm}")
        plt.plot([i*90 for i in range(len(data_total[key][algorithm]['pred']))], data_total[key][algorithm]['pred'], label="prediction")
        plt.plot(data_total[key][algorithm]['label'], label="ground truth")
        plt.legend()
        plt.show()

