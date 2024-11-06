import os
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def str2num(data):
    try:
        first = float(data.split('e')[0])
        second = float(data.split('e')[1])
        return first * 10**(second)
    except ValueError:
        return float(data)
    except IndexError:
        return float(data)
    

base_dir = "D:/home/BCML/drax/PAPER/data/rppg_acc_synchro/"
video_data_base_dir = "D:/home/BCML/drax/PAPER/data/results/rppg_emma/"
save_dir = "D:/home/BCML/drax/PAPER/data/results/refined/"

data_folder_list = os.listdir(video_data_base_dir)

for folder in data_folder_list:
    temp_bvp = {}
    temp_bvp['value'] = []
    temp_bvp['index'] = []
    
    with open(video_data_base_dir+folder+'/ppg_omit.txt', 'r') as f:
        lines = f.readlines()
        temp_bvp_step = []
        for line in lines:
            if '[' in line:
                temp_line = line.strip('[')
                if temp_line[0] == ' ':
                    temp_line = temp_line[1:]
                temp_values = temp_line.split(' ')
                temp_values = [value for value in temp_values if value != '' and value != ' ']
                temp_append = [str2num(value) for value in temp_values]
                temp_bvp_step.append(temp_append)
            elif '[' not in line and ']' not in line:
                temp_values = temp_line.split(' ')
                temp_values = [value for value in temp_values if value != '' and value != ' ']
                temp_append = [str2num(value) for value in temp_values]
                temp_bvp_step.append(temp_append)
            elif ']' in line:
                temp_line = line.split(']')[0]
                temp_values = temp_line.strip(' ').split(' ')
                temp_values = [value for value in temp_values if value != '']
                temp_append = [str2num(value) for value in temp_values]
                temp_bvp_step.append(temp_append)
                temp_bvp['value'].append(temp_bvp_step)
                temp_bvp_step = []
                temp_bvp['index'].append(int(line.split(']')[1]))
    
    final_bvp = []
    for i in range(len(temp_bvp['value'])):
        final_bvp.append(stats.trim_mean(temp_bvp['value'][i][-1], 0.3))
    
    plt.plot(final_bvp)
    plt.show()
    final_bvp = np.array(final_bvp)
    np.save(f"{save_dir}{folder[:5]}.npy", final_bvp)
    print(f"{folder[:5]}.npy is saved")