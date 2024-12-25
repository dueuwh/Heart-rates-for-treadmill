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

def load_dataset_3():
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
    
    return data_total, algorithms

def load_dataset():
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
    
    return data_total, algorithms

class freq_model():
    def __init__(self, hyperparameter_setting):
    
        self.num_window = hyperparameter_setting["num_window"]
        self.num_step = hyperparameter_setting["num_step"]
        self.num_fme = hyperparameter_setting["num_frequency_mean_ewma"]
        self.num_fse = hyperparameter_setting["num_frequency_smoothing_ewma"]
        self.num_fde = hyperparameter_setting["num_frequency_diff_ewma"]
        self.num_bpm_ewma = hyperparameter_setting["num_bpm_ewma"]
        
        self.a4fw = hyperparameter_setting["alpha_4_frequency_window"]
        self.a4fdm = hyperparameter_setting["alpha_4_frequency_diff_main"]
        self.a4fde = hyperparameter_setting["alpha_4_frequency_diff_end"]
        self.a4fds = hyperparameter_setting["alpha_4_frequency_diff_start"]
        self.a4bo = hyperparameter_setting["alpha_4_bpm_output"]
        
        self.exp_base_curve = hyperparameter_setting["exp_base_curve"]
        self.exp_power_curve = hyperparameter_setting["exp_power_curve"]
        self.exp_base_scaling = hyperparameter_setting["exp_base_scaling"]
        self.exp_power_scaling = hyperparameter_setting["exp_power_scaling"]
        self.bias = hyperparameter_setting["bias"]
        
        self.list_window = []
        self.list_freq_ewma = []
        self.list_diff_ewma = []
        self.list_bpm_for_step = []
        self.bpm_previous = 0
        self.scaling = hyperparameter_setting["scaling_factor"]
        
        self.sig_p_coeffi = hyperparameter_setting["sigmoid_p_coefficient"]
        self.sig_e_const = hyperparameter_setting["sigmoid_e_constant"]
        self.sig_p_const = hyperparameter_setting["sigmoid_p_constant"]
        self.sig_e_numerator = hyperparameter_setting["sigmoid_e_numerator"]
        self.sig_const = hyperparameter_setting["sigmoid_constant"]
        self.arousal_adapt = hyperparameter_setting["arousal_adapt"]
        
        self.bpm_delay_window = hyperparameter_setting["bpm_delay_window"]
        
        self.exp_power_curve_idx = 0
        self.run_count = 0
        
        self.bpm_queue = []
        
        # All exception is disalbed for BOHB test
        #
        # if not 0<self.a4fw<1 or not 0<self.a4fdm<1 or not 0<self.a4fde<1 or not 0<self.a4fds<1:
        #     raise ValueError("value alpha must be in the range : 0 < alpha < 1")
            
        # if exp_power_curve>=0 or exp_power_scaling>=0:
        #     raise ValueError("power of exponential function must be smaller than 0")
    
    def arithmetic_mean(self, input_series):
        return sum(input_series)/len(input_series)
    
    def ewma_forward(self, input_series, alpha=0.5):
        process_series = input_series
        ewma_output = process_series[0]
        if len(process_series)>1:
            for idx in range(len(process_series)-1):
                ewma_output = alpha * ewma_output + (1-alpha) * process_series[idx+1]
        else:
            pass
        return ewma_output

    def ewma_backward(self, input_series, alpha=0.5):
        process_series = input_series
        ewma_output = process_series[-1]
        if len(process_series)>1:
            for idx in range(len(process_series)-1):
                ewma_output = alpha * ewma_output + (1-alpha) * process_series[-idx-1]
        else:
            pass
        return ewma_output
    
    def harmonic_mean(self, input_series):
        denominator = 0
        for value in input_series:
            denominator += (1/value)
        return len(input_series)/denominator
    
    def geometric_mean(self, input_series):
        return output
    
    def harmonic_mean_ewma(self, input_series, alpha=0.5):
        inverse_list = [1/value for value in input_series]
        return len(input_series)/(self.ewma_forward(inverse_list))
    
    def geometric_mean_ewma(self, input_series, alpha=0.5):
        return output
    
    def extract_feature_freq(self, inputs):
        inputs_sorted = list(deepcopy(inputs))
        inputs_sorted.sort(reverse=True)
        inputs_rank = []
        for i in range(len(inputs)):
            inputs_rank.append(inputs_sorted.index(inputs[i]))
        
        feature_freq_list = []
        for i in range(self.num_fme):
            feature_freq_list.append(inputs_rank.index(i))
        
        feature_freq_out = self.arithmetic_mean(feature_freq_list)  # mean of frequency list : futher work: To use statistical method
        
        return feature_freq_out
    
    def sigmoid(self, x):
        return self.sig_numer/(1+math.exp(self.sig_coeffi*(x+self.sig_const)))
    
    def delay(self, x):
        if len(self.bpm_queue) >= self.bpm_delay_window:
            output = sum(self.bpm_queue)/self.bpm_delay_window
            self.bpm_queue.pop(0)
            return output
        else:
            return x
        
    
    def run(self, input_freq, est_start, est_start_initial_bpm=None):
        
        self.run_count += 1
        
        if est_start:
            if est_start_initial_bpm == None:
                raise ValueError("initial BPM is 'None'")
            self.bpm_previous = est_start_initial_bpm
        
        feature_freq = self.extract_feature_freq(input_freq)
        # feature_freq = input_freq
        
        if len(self.list_freq_ewma) == self.num_fse:
            del self.list_freq_ewma[0]
        
        self.list_freq_ewma.append(feature_freq)
        self.list_window.append(self.ewma_forward(self.list_freq_ewma, self.a4fw))
        
        if len(self.list_window) == self.num_window:
            
            diff_end = self.ewma_forward(self.list_window, self.a4fde)
            diff_start = self.ewma_backward(self.list_window, self.a4fds)
            diff_acceleration = diff_end - diff_start

            diff_main = self.ewma_forward(self.list_window, self.a4fdm)
            diff = diff_acceleration * diff_main
            
            if len(self.list_diff_ewma) == self.num_fde:
                del self.list_diff_ewma[0]
            
            self.list_diff_ewma.append(diff)
            diff_final = self.ewma_forward(self.list_diff_ewma)
            
            # diff_sign = self.sig_e_numerator/(np.exp(self.sig_p_const+self.sig_p_coeffi*diff_end)+self.sig_e_const)-self.sig_const
            
            # Utilizing an exponential function to progressively reduce the rate of change in BPM over frequency
            bpm_output = self.bpm_previous + diff_final * self.scaling * self.exp_base_scaling ** (self.exp_power_scaling * abs(diff_final)) + self.bias
            # bpm_output = self.bpm_previous + diff_sign * diff_final * self.scaling * self.exp_base_scaling ** (self.exp_power_scaling * abs(diff_final)) + self.bias
            # Utilizing an sigmoid function to progressively reduce the rate of change in BPM over frequency
            # bpm_output = self.bpm_previous + diff_final * self.scaling * self.sigmoid(abs(diff_final))
                  
            del self.list_window[0]
            
            self.bpm_previous = bpm_output
            
            self.bpm_queue.append(bpm_output)
            # output = self.delay(bpm_output)
            
            return bpm_output

def fft_half(sig_window, sampling_rate, n_fft):
    s_fft = np.fft.fft(sig_window)
    ampl = abs(s_fft) * (2 / len(s_fft))
    f = np.fft.fftfreq(len(s_fft), 1 / sampling_rate)
    
    temp_normalization = []
    for value in ampl:
        if value > 0:
            temp_normalization.append(math.log10(value))
        else:
            temp_normalization.append(0.0)
    
    ampl_out = np.array(temp_normalization)

    ampl_out = ampl_out[0:int(n_fft // 2)]
    f = f[0:int(n_fft // 2)]
    return ampl_out, f

def fft_half_normalization_test(sig_window, sampling_rate, n_fft):
    s_fft = np.fft.fft(sig_window)
    ampl = abs(s_fft) * (2 / len(s_fft))
    f = np.fft.fftfreq(len(s_fft), 1 / sampling_rate)
    
    temp_normalization = []
    for value in ampl:
        temp_normalization.append(value)
    
    ampl_out = np.array(temp_normalization)

    ampl_out = ampl_out[0:int(n_fft // 2)]
    f = f[0:int(n_fft // 2)]
    return ampl_out, f

def compute_fft(signal, sampling_freq=30):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_freq)[:N//2]
    yf = 2.0/N * np.abs(yf[0:N//2])
    return xf, yf

if __name__ == "__main__":
    path = "D:/home/BCML/IITP/rPPG-Toolbox/evaluation/post_process.py"
    function = "_calculate_fft_hr"
    cal_hr_fft = import_function_from_file(path, function)
    
    ceemdan = CEEMDAN()
    speed_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/video_speed.txt"
    speeds = {}
    with open(speed_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            speeds[line.split('.')[0]] = []
            speed_info = line.split(":")[1].split(',')
            speed_info = [value for value in speed_info if len(value) > 1]
            for speed in speed_info:
                speeds[line.split('.')[0]].append([int(speed.split('_')[0]), int(speed.split('_')[1])])
    
    total_data, algorithms = load_dataset_3()
    
    index = [0,3,5,7,9,'total']
    columns = deepcopy(algorithms)
    for algorithm in algorithms:
        columns.append(f'Frequency_{algorithm}')
    final_table_mae = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_rmse = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_r2 = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count4r2 = pd.DataFrame(0.0, index=index, columns=columns)
    
    for num in range(23): 
        input_num = label_num = num
        coords_path = "D:/home/BCML/drax/PAPER/data/coordinates/stationary_kalman_2nd/"
        coords_folders = os.listdir(coords_path)
        coords_input = np.load(coords_path+coords_folders[input_num])
        coords_input_x = coords_input[:, 0]
        coords_input_y = coords_input[:, 1]
        mean = np.mean(coords_input_x)
        std = np.std(coords_input_x)
        coords_input_x = (coords_input_x - mean) / std
        coords_input_detrend = np.diff(coords_input_x, 1)
        mean = np.mean(coords_input_y)
        std = np.std(coords_input_y)
        coords_input_y = (coords_input_y - mean) / std
        coords_input_y_detrend = np.diff(coords_input_y, 1)
        
        model = freq_model(Params.frequency_model_hyperparameters)
        
        fft_window = 180
        
        pass_band = (0.5, 10)
        biquad_section = 10
        sampling_rate = 30  # video FPS
        sos = signal.butter(biquad_section, pass_band, 'bandpass', fs=sampling_rate, output='sos')
        
        total_data_keys = [*total_data.keys()]
        input_data = total_data[total_data_keys[label_num]]
        
        for al_index, algorithm in enumerate(algorithms):
            
            os.makedirs(f"./materials/fft_version_6sec/{algorithm}", exist_ok=True)
            
            bvp = input_data[algorithm]['pred']
            x = []
            for i in range((len(bvp)-fft_window)):
                x.append(cal_hr_fft(bvp[i:i+fft_window]))
            y = input_data[algorithm]['label']
            
    
            if len(coords_input_y_detrend) > len(y):
                original_indices = np.linspace(0, len(coords_input_y_detrend), len(coords_input_y_detrend))
                target_indices = np.linspace(0, len(y), len(y))
                interpolator = interp1d(original_indices, coords_input_y_detrend, kind='linear')
                coords_input_y_detrend = interpolator(target_indices)
            else:
                target_indices = np.linspace(0, len(coords_input_y_detrend), len(coords_input_y_detrend))
                original_indices = np.linspace(0, len(y), len(y))
                interpolator = interp1d(original_indices, y, kind='linear')
                y = interpolator(target_indices)
            
            total_results = []
            ampl_input_list = []
            for i in range(len(coords_input_y_detrend)-fft_window):
                ldmk_bpf = signal.sosfilt(sos, coords_input_y_detrend[i:i+fft_window])
                
                ampl_input, _ = fft_half(ldmk_bpf, sampling_rate, fft_window)
                ampl_input_list.append(ampl_input[0])
                
                # IMFs = ceemdan.ceemdan(ldmk_bpf)
                # for i, imf in enumerate(IMFs):
                #     xf, yf = compute_fft(imf)
                #     plt.figure(figsize=(12, 3))
                #     plt.plot(xf, yf)
                #     plt.scatter(xf[np.where(yf==max(yf))], max(yf), s=10, c='r')
                #     plt.title(f'IMF {i+1} Frequency Spectrum')
                #     plt.xlabel('Frequency [Hz]')
                #     plt.ylabel('Amplitude')
                #     plt.grid()
                #     plt.show()
                
                # try:
                #     xf, yf = compute_fft(IMFs[3])
                # except IndexError:
                #     try:
                #         xf, yf = compute_fft(IMFs[2])
                #     except IndexError:
                #         xf, yf = compute_fft(IMFs[1])
                
                # ampl_input = xf[np.where(yf==max(yf))][0]
                # ampl_input_list.append(ampl_input)
                
                if i == 0:
                    run_start = True
                    initial_bpm = x[0]
                    est_bpm = model.run(ampl_input, run_start, initial_bpm)
                else:
                    run_start = False
                    est_bpm = model.run(ampl_input, run_start)
                
                if est_bpm is not None:
                    total_results.append(est_bpm)
                else:
                    if len(total_results) >= 1:
                        total_results.append(total_results[-1])
                    else:
                        total_results.append(0)
                    
            if al_index == 0:
                interval_freq = len(y)/len(ampl_input_list)
                plt.figure(figsize=(10, 6))
                plt.title(f"{coords_folders[input_num]} STFT")
                plt.plot([i*interval_freq for i in range(len(ampl_input_list))], ampl_input_list, label="STFT")
                
                plt.savefig(f"./materials/fft_version_6sec/{coords_folders[input_num]}_stft_input.png")
                plt.show()
                
            interval_rppg = len(y)/len(x)
            
            print("len(x): ", len(x))
           
            new_y_interval = len(y)/len(x)
            y = [y[int(i*new_y_interval)] for i in range(len(x))]
            print("len(y): ", len(y))
            
            new_total_results_interval = len(total_results)/len(x)
            total_results_temp = []
            for i in range(len(x)):
                temp_index = int(i*new_total_results_interval)
                temp_value = total_results[temp_index]
                
                if i==0 and temp_value == np.nan:
                    while True:
                        temp_index += 1
                        temp_value = total_results[temp_index]
                        if temp_value != np.nan:
                            total_results_temp.append(temp_value)
                            break
                else:
                    while True:
                        temp_index -= 1
                        temp_value = total_results[temp_index]
                        if temp_value != np.nan:
                            total_results_temp.append(temp_value)
                            break
                    
            total_results = total_results_temp
            print("len(total_results): ", len(total_results))
            
            plt.figure(figsize=(12,6))
            plt.title(f"{coords_folders[input_num]} {algorithm}")
            plt.plot([i*new_total_results_interval for i in range(1,len(x))], total_results[1:], label="frequency model pred")
            plt.plot([i*new_y_interval for i in range(len(x))], y, label="ground truth")
            plt.plot([i*interval_rppg for i in range(len(x))], x, label="rppg pred")
            temp_value = speeds[coords_folders[input_num].split('.')[0]]
            plt.text(temp_value[0][1]-200, 70, 0, ha='center', va='bottom', fontsize=15, color='red')
            for value in temp_value:
                plt.axvline(value[1], 0, 200, color='red', linestyle='--', linewidth=2)
                plt.text(value[1]+80, 70, value[0], ha='center', va='bottom', fontsize=20, color='red')
            # plt.plot([i*interval_freq for i in range(len(ampl_input_list))], ampl_input_list, label="STFT")
            # plt.plot(coords_input_y_detrend, label="input data")
            plt.legend()
            plt.savefig(f"./materials/fft_version_6sec/{algorithm}/{coords_folders[input_num]}_performance.png")
            plt.show()

            #==========================================================================================

            x_acc_mae = mean_absolute_error(y[1:], x[1:])
            x_acc_rmse = np.sqrt(mean_squared_error(y[1:], x[1:]))
            x_acc_r2 = r2_score(y[1:], x[1:])
            
            with open(f"./materials/fft_version_6sec/{algorithm}/performance.txt", 'a') as f:
                f.write(f"{coords_folders[input_num]} x:mae({x_acc_mae})rmse({x_acc_rmse})r2({x_acc_r2})\n")
            
            final_table_mae.loc['total', algorithm] += x_acc_mae
            final_table_rmse.loc['total', algorithm] += x_acc_rmse
            if 0 <= x_acc_r2 <= 1:
                final_table_r2.loc['total', algorithm] += x_acc_r2
                final_table_count4r2.loc['total', algorithm] += 1
            final_table_count.loc['total', algorithm] += 1
            
            new_temp_value = []
            for value in temp_value:
                new_temp_value.append([value[0], int(value[1]//new_y_interval)])
            
            with open(f"./materials/fft_version_6sec/{algorithm}/{coords_folders[input_num]}_time_index.txt", 'a') as f:
                for value in new_temp_value:
                    f.write(f"{value[0]}_{value[1]}\n")
            
            previous_point = 0
            for value in new_temp_value:
                x_acc_mae = mean_absolute_error(y[previous_point:value[1]], x[previous_point:value[1]])
                x_acc_rmse = np.sqrt(mean_squared_error(y[previous_point:value[1]], x[previous_point:value[1]]))
                x_acc_r2 = r2_score(y[previous_point:value[1]], x[previous_point:value[1]])
                previous_point = value[1]
                
                with open(f"./materials/fft_version_6sec/{algorithm}/performance.txt", 'a') as f:
                    f.write(f"{coords_folders[input_num]}speed({value[0]}) x:mae({x_acc_mae})rmse({x_acc_rmse})r2({x_acc_r2})\n")
                
                final_table_mae.loc[int(value[0]), algorithm] += x_acc_mae
                final_table_rmse.loc[int(value[0]), algorithm] += x_acc_rmse
                if 0 <= x_acc_r2 <= 1:
                    final_table_r2.loc[int(value[0]), algorithm] += x_acc_r2
                    final_table_count4r2.loc[int(value[0]), algorithm] += 1
                final_table_count.loc[int(value[0]), algorithm] += 1
                
            #==========================================================================================
            
            total_results_acc_mae = mean_absolute_error(y[1:], total_results[1:])
            total_results_acc_rmse = np.sqrt(mean_squared_error(y[1:], total_results[1:]))
            total_results_acc_r2 = r2_score(y[1:], total_results[1:])
            
            with open(f"./materials/fft_version_6sec/{algorithm}/performance.txt", 'a') as f:
                f.write(f"{coords_folders[input_num]} freq:mae({total_results_acc_mae})rmse({total_results_acc_rmse})r2({total_results_acc_r2})\n")

            final_table_mae.loc['total', f'Frequency_{algorithm}'] += total_results_acc_mae
            final_table_rmse.loc['total', f'Frequency_{algorithm}'] += total_results_acc_rmse
            if 0 <= total_results_acc_r2 <= 1:
                final_table_r2.loc['total', f'Frequency_{algorithm}'] += total_results_acc_r2
                final_table_count4r2.loc['total', f'Frequency_{algorithm}'] += 1
            final_table_count.loc['total', f'Frequency_{algorithm}'] += 1

            previous_point = 0
            
            for value in new_temp_value:
                total_results_acc_mae = mean_absolute_error(y[previous_point:value[1]], total_results[previous_point:value[1]])
                total_results_acc_rmse = np.sqrt(mean_squared_error(y[previous_point:value[1]], total_results[previous_point:value[1]]))
                total_results_acc_r2 = r2_score(y[previous_point:value[1]], total_results[previous_point:value[1]])
                previous_point = value[1]
                
                with open(f"./materials/fft_version_6sec/{algorithm}/performance.txt", 'a') as f:
                    f.write(f"{coords_folders[input_num]}speed({value[0]}) freq:mae({total_results_acc_mae})rmse({total_results_acc_rmse})r2({total_results_acc_r2})\n")
                
                final_table_mae.loc[int(value[0]), f'Frequency_{algorithm}'] += total_results_acc_mae
                final_table_rmse.loc[int(value[0]), f'Frequency_{algorithm}'] += total_results_acc_rmse
                if 0 <= total_results_acc_r2 <= 1:
                    final_table_r2.loc[int(value[0]), algorithm] += total_results_acc_r2
                    final_table_count4r2.loc[int(value[0]), algorithm] += 1
                final_table_count.loc[int(value[0]), f'Frequency_{algorithm}'] += 1
                
            np.save(f"./materials/fft_version_6sec/{algorithm}/{coords_folders[input_num]}_y.npy", np.array(y))
            np.save(f"./materials/fft_version_6sec/{algorithm}/{coords_folders[input_num]}_x.npy", np.array(x))
            np.save(f"./materials/fft_version_6sec/{algorithm}/{coords_folders[input_num]}_total_results.npy", np.array(total_results))
            np.save(f"./materials/fft_version_6sec/{algorithm}/{coords_folders[input_num]}_ampl_input_list.npy", np.array(ampl_input_list))
                
    final_table_mae = final_table_mae/final_table_count
    final_table_rmse = final_table_rmse/final_table_count
    final_table_r2 = final_table_r2/final_table_count4r2
    
    final_table_mae.to_excel(f"./materials/fft_version_6sec/mae_table.xlsx")
    final_table_rmse.to_excel(f"./materials/fft_version_6sec/rmse_table.xlsx")
    final_table_r2.to_excel(f"./materials/fft_version_6sec/r2_table.xlsx")
    
    ax = sns.heatmap(final_table_mae)
    plt.title("total mae")
    plt.savefig(f"./materials/fft_version_6sec/mae_table.png")
    plt.show()
    
    ax = sns.heatmap(final_table_rmse)
    plt.title("total rmse")
    plt.savefig(f"./materials/fft_version_6sec/rmse_table.png")
    plt.show()
    
    ax = sns.heatmap(final_table_r2, vmin=0.0, vmax=1.0)
    plt.title("total r2")
    plt.savefig(f"./materials/fft_version_6sec/r2_table.png")
    plt.show()
# legacy

# num = 0

# video_path = "C:/Users/U/Desktop/BCML/Drax/Videos/treadmill_data/"
# ref_path = "C:/Users/U/Desktop/BCML/Drax/refs_interpolated/"

# video_list = os.listdir(video_path)
# ref_list = os.listdir(ref_path)

# sel_video = video_list[num]
# sel_ref = ref_list[num]

# video_cap = cv2.VideoCapture(video_path+sel_video)
# ref = pd.read_csv(ref_path+sel_ref, index_col=0)['BPM'].values

# ref_list = [name.split('.')[0]+'.xlsx' for name in video_list]
# num_fft = 300
# sampling_rate = 30
# initial_bpm = ref[num_fft]

# base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
# options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                        output_face_blendshapes=False,
#                                        output_facial_transformation_matrixes=True,
#                                        num_faces=1)
# detector = vision.FaceLandmarker.create_from_options(options)

# face_landmarks_list = [151, 9, 8, 168, 5, 4, 1, 0, 18, 101, 330, 105, 334, 208, 428, 107, 336, 411, 147, 2]
# len_list = len(face_landmarks_list)
# fft_arr_x = []
# fft_arr_y = []
# fft_arr_all = np.zeros((num_fft, 2*len_list))
# input_freq = np.zeros((2*len_list, num_fft//2))

# temp_result = []
# index = 0
# while True:
#     ret, frame = video_cap.read()
#     if not ret:
#         print(f"{sel_video} end")
#         break
#     frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#     detection_result = detector.detect(frame)
#     face_landmarks = detection_result.face_landmarks[0]
    
#     temp_x = np.zeros((len_list))
#     temp_y = np.zeros((len_list))
    
#     for idx in range(len(face_landmarks_list)):
#         temp_x[idx] = face_landmarks[face_landmarks_list[idx]].x
#         temp_y[idx] = face_landmarks[face_landmarks_list[idx]].y
    
#     if len(fft_arr_x) == num_fft:
#         del fft_arr_x[0]
#         del fft_arr_y[0]
    
#     fft_arr_x.append(temp_x)
#     fft_arr_y.append(temp_y)
    
#     if len(fft_arr_x) == num_fft:
        
#         for i in range(num_fft):
#             fft_arr_all[i,0:len_list] = fft_arr_x[i][:]
#             fft_arr_all[i,len_list:] = fft_arr_y[i][:]
        
#         for i in range(len_list*2):
#             input_freq[i] = fft_half(fft_arr_all[:, i], num_fft, sampling_rate)[:]
        
#         input_freq_mean = [sum(input_freq[:, i])/len(input_freq[:, i]) for i in range(num_fft//2)]
#         input_freq_mean = input_freq_mean[2:]
        
#         if index == 0:
#             temp_result.append(model.run(input_freq_mean, True, initial_bpm))
#         else:
#             temp_result.append(model.run(input_freq_mean, False))
#         index += 1
#         print(index)