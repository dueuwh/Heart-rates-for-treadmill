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
from scipy.stats import pearsonr
import sys
import seaborn as sns
from PyEMD import CEEMDAN
from scipy.fft import fft, fftfreq
import importlib.util

def find_good_index(x, y, speed3_start, error_threshold=10.0, verbose=False):
    for i in range(len(x)+180-speed3_start):
        error = abs(y[i+speed3_start] - x[i-180+speed3_start])
        if error < error_threshold:
            if verbose:
                print("error: ", error)
                print("good_index: ", i+speed3_start)
                plt.figure(figsize=(12,6))
                plt.title(f"x: {x[i+speed3_start-180]} y: {y[i+speed3_start]}")
                plt.plot([i+180 for i in range(len(x))], x, label="x", zorder=2)
                plt.plot(y, label='y', zorder=3)
                plt.axvline(i+speed3_start, 0, 200, c='r', zorder=1)
                plt.axvline(speed3_start, 0, 200, c='g', zorder=0)
                plt.legend()
                plt.show()
            
            return i+speed3_start
    return None

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def import_function_from_file(file_path, function_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def minmax(series):
    min_value = min(series)
    max_value = max(series)
    
    denominator = max_value-min_value
    
    output = []
    for value in series:
        output.append((value-min_value)/denominator/10)
    return output

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
        
        self.acc_length = hyperparameter_setting["acc_list_length"]
        self.acc_queue = []
        self.acc_sign_length = hyperparameter_setting["acc_sign_list_length"]
        self.acc_sign_queue = []
        
        self.exp_power_curve_idx = 0
        self.run_count = 0
        
        self.diff_main_length = hyperparameter_setting["diff_main_list_length"]
        self.diff_main_list = []
        
        self.diff_coeff_scaling = hyperparameter_setting["diff_coeff_scaling"]
        
        self.diff_coeff_e_coeff = hyperparameter_setting["diff_coeff_e_coeff"]
        self.diff_coeff_e_constant = hyperparameter_setting["diff_coeff_e_constant"]
        
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
            
            #=================================================================================
            # cal diff main  
            #=================================================================================

            diff_end = self.ewma_forward(self.list_window, self.a4fde)
            diff_start = self.ewma_backward(self.list_window, self.a4fds)
            

            diff_main = self.ewma_forward(self.list_window, self.a4fdm)
            self.diff_main_list.append(diff_main)
            
            if len(self.diff_main_list) > self.diff_main_length:
                del self.diff_main_list[0]

            #=================================================================================
            # cal diff acc
            #=================================================================================

            if len(self.diff_main_list) > 0:
                diff_acceleration = np.mean(self.diff_main_list)
            else:
                diff_acceleration = 0

            self.acc_queue.append(diff_acceleration)
            
            if len(self.acc_queue) > self.acc_length:
                del self.acc_queue[0]
            
            #=================================================================================
            # cal diff acc sign
            #=================================================================================
            
            if len(self.acc_queue) > 2:
                diff_acc_sign = np.mean(np.diff(self.acc_queue))
            else:
                diff_acc_sign = 0

            self.acc_sign_queue.append(diff_acc_sign)
            
            if len(self.acc_sign_queue) > self.acc_sign_length:
                del self.acc_sign_queue[0]
            
            #=================================================================================
            # cal diff acc sign coefficient
            #=================================================================================
            
            if len(self.acc_sign_queue) > 0:
                diff_acc_sign_coefficient = np.mean(self.acc_sign_queue)
            else:
                diff_acc_sign_coefficient = 0
            
            diff_final = diff_acc_sign_coefficient * diff_acceleration
            
            # diff_sign = self.sig_e_numerator/(np.exp(self.sig_p_const+self.sig_p_coeffi*diff_end)+self.sig_e_const)-self.sig_const
            
            # Original
            # bpm_output = self.bpm_previous + diff_final * self.scaling * self.exp_base_scaling ** (self.exp_power_scaling * abs(diff_final)) + self.bias
            bpm_output = self.bpm_previous + diff_final * self.scaling
            
  
            del self.list_window[0]
            
            self.bpm_previous = bpm_output
            
            self.bpm_queue.append(bpm_output)
            # output = self.delay(bpm_output)
            return bpm_output, diff_acceleration, diff_acc_sign, diff_acc_sign_coefficient, diff_final, diff_main
        else:
            return self.bpm_previous, 0, 0, 0, 0, 0


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
    start_good_index = False
    good_index = 0
    
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
    
    total_data, algorithms = load_dataset()
    
    index = [0,3,5,7,9,'total']
    columns = deepcopy(algorithms)
    for algorithm in algorithms:
        columns.append(f'Frequency_{algorithm}')
    final_table_mae = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_rmse = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_r2 = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count4r2 = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_r2_adjust = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count4r2_adjust = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_pearson = pd.DataFrame(0.0, index=index, columns=columns)
    columns4spectrogram_pearson = ["total"]
    final_table_spectrogram_pearson = pd.DataFrame(0.0, index=index, columns=columns4spectrogram_pearson)
    final_table_spectrogram_pearson_count = pd.DataFrame(0.0, index=index, columns=columns4spectrogram_pearson)
    
    for num in range(23): 
        for al_index, algorithm in enumerate(algorithms):
            
            os.makedirs(f"./materials/fft_version_6_sec/{algorithm}", exist_ok=True)
            
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
            
            words4title = coords_folders[input_num].split('_')
            title_plot = f'{words4title[0]}_{words4title[1]}'
            
            if al_index == 0:
                plt.figure(figsize=(12,6))
                plt.title(f"{title_plot} preprocessed data")
                plt.plot(coords_input_y_detrend, label="input data")
                plt.legend()
                plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_input_data_plot.png")
                plt.show()
                
                plt.figure(figsize=(12,6))
                plt.title(f"{title_plot} raw data")
                plt.plot(coords_input[:, 1], label="raw data")
                plt.legend()
                plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_raw_data_plot.png")
                plt.show()
                
                frequencies, times, Sxx = signal.spectrogram(coords_input_y_detrend, 30.0)
                plt.figure(figsize=(12, 6))
                plt.title(f"{title_plot} spectrogram")
                plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.colorbar(label='Intensity [dB]')
                plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_spectrogram.png")
                plt.show()
                
                max_indices = np.argmax(Sxx, axis=0)
                max_frequencies = frequencies[max_indices]
                plt.figure(figsize=(10, 6))
                plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
                plt.plot(times, max_frequencies, color='red', linewidth=2, label='Max Amplitude Frequency')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.title(f'{title_plot} spectrogram with maximum amplitude frequencies')
                plt.colorbar(label='Intensity [dB]')
                plt.legend()
                plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_spectrogram_with_max_frequency.png")
                plt.show()
            
            
            model = freq_model(Params.frequency_model_hyperparameters)
            
            fft_window = 180
            
            pass_band = (0.5, 10)
            biquad_section = 10
            sampling_rate = 30  # video FPS
            sos = signal.butter(biquad_section, pass_band, 'bandpass', fs=sampling_rate, output='sos')
            
            total_data_keys = [*total_data.keys()]
            input_data = total_data[total_data_keys[label_num]]
            
            bvp = input_data[algorithm]['pred']
            x = []
            for i in range((len(bvp)-fft_window)):
                x.append(cal_hr_fft(bvp[i:i+fft_window]))
            y = input_data[algorithm]['label']
            
            total_results = []
            ampl_input_list = []
            acc_log = []
            acc_sign_log = []
            diff_acc_sign_coefficient_log = []
            diff_final_log = []
            diff_main_log = []
            
            # start at speed 3
            # crop if y length is longer than x length + 180
            if len(x)+180 <= len(y):
                crop_index = -(len(y)-len(x)-180)
                y = y[:crop_index]
                coords_input_y_detrend = coords_input_y_detrend[:crop_index]
            # crop if x length + 180 is longer than length of y
            elif len(x)+180 >= len(y):
                crop_index = -(len(x)+180-len(y))
                x = x[:crop_index]
            
            speed3_start = speeds[coords_folders[input_num].split('.')[0]][0][1]
            speed3_end = None
            if len(speeds[coords_folders[input_num].split('.')[0]]) >= 5:
                speed3_end = speeds[coords_folders[input_num].split('.')[0]][-1][1]

            coords_input_y_detrend = coords_input_y_detrend[speed3_start:]
            
            # find bpm index for maximize accuracy of my algorithm
            if start_good_index:
                good_index = find_good_index(x, y, speed3_start)
                if good_index == None:
                    continue
            
            # set initial bpm
            initial_point = speed3_start
            if start_good_index:
                initial_point = good_index - 180
                coords_input_y_detrend_temp = deepcopy(coords_input_y_detrend[(good_index-speed3_start):])
            else:
                coords_input_y_detrend_temp = coords_input_y_detrend
            
            
            #==============================================================================
            # Run my algorithm
            #==============================================================================
            for i in range(len(coords_input_y_detrend_temp)-fft_window):
                ldmk_bpf = signal.sosfilt(sos, coords_input_y_detrend_temp[i:i+fft_window])
                
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
                    initial_bpm = x[initial_point]
                    est_bpm, acc, acc_sign, diff_acc_sign_coefficient, diff_final, diff_main = model.run(ampl_input, run_start, initial_bpm)
                else:
                    run_start = False
                    est_bpm, acc, acc_sign, diff_acc_sign_coefficient, diff_final, diff_main = model.run(ampl_input, run_start)
                
                acc_log.append(acc)
                acc_sign_log.append(acc_sign)
                diff_acc_sign_coefficient_log.append(diff_acc_sign_coefficient)
                diff_final_log.append(diff_final)
                diff_main_log.append(diff_main)
                
                if est_bpm is not None:
                    total_results.append(est_bpm)
                else:
                    if len(total_results) >= 1:
                        total_results.append(total_results[-1])
                    else:
                        total_results.append(0)
            
            
            #==============================================================================
            # drawing figures for paper
            #==============================================================================
            if al_index == 0:
                interval_freq = len(y)/len(ampl_input_list)
                plt.figure(figsize=(12, 6))
                plt.title(f"{coords_folders[input_num]} STFT")
                plt.plot([i*interval_freq for i in range(len(ampl_input_list))], ampl_input_list, label="STFT")
                plt.savefig(f"./materials/fft_version_6_sec/{coords_folders[input_num]}_stft_input.png")
                plt.show()
                
                plt.figure(figsize=(12, 6))
                plt.title(f"{title_plot} heart rate label")
                plt.plot(y, label="heart rate")
                plt.legend()
                plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_label.png")
                plt.show()
                
                y4freq_pearson_interval = len(y)/len(max_frequencies)
                y4freq_pearson = [y[int(i*y4freq_pearson_interval)] for i in range(len(max_frequencies))]
                dict4pearson = {
                    "max frequency":max_frequencies,
                    "Label":y4freq_pearson}
                df4pearson = pd.DataFrame(dict4pearson)
                corr_matrix = df4pearson.corr(method='pearson')
                final_table_spectrogram_pearson.loc['total', 'total'] += corr_matrix.loc["max frequency", "Label"]
                final_table_spectrogram_pearson_count.loc['total', 'total'] += 1
                
                previous_point = 0
                for value_set in speeds[coords_folders[input_num].split('.')[0]]:
                    speed = value_set[0]
                    index = value_set[1]
                    
                    index = int(index/y4freq_pearson_interval)
                    
                    dict4pearson = {
                        "max frequency":max_frequencies[previous_point:index],
                        "Label":y4freq_pearson[previous_point:index]}
                    df4pearson = pd.DataFrame(dict4pearson)
                    corr_matrix = df4pearson.corr(method='pearson')
                    final_table_spectrogram_pearson.loc[speed, "total"] += corr_matrix.loc["max frequency", "Label"]
                    final_table_spectrogram_pearson_count.loc[speed, "total"] += 1
            #==============================================================================
            
            #==============================================================================
            # Matching x, y length for accuracy calculation
            #==============================================================================
            y4_acc = []
            if start_good_index:
                if good_index != 0:
                    y4_acc = y[good_index:]
                else:
                    y4_acc = y[speed3_start:]
            else:
                y4_acc = y[speed3_start:]
            if len(y4_acc) > len(total_results):
                y4_acc = y4_acc[:-(len(y4_acc)-len(total_results))]
            
            if start_good_index:
                if good_index != 0:
                    x4_acc = x[good_index-180:]
                else:
                    x4_acc = x[speed3_start-180:]
            else:
                x4_acc = x[speed3_start-180:]
            if len(x4_acc) > len(total_results):
                x4_acc = x4_acc[:-(len(x4_acc)-len(total_results))]
            
            #==============================================================================
            # Matching x, y length if those datasets are longer datasets
            #==============================================================================
            
            if speed3_end != None:
                tail_crop_y = len(y4_acc)-(speed3_end-speed3_start)
                tail_crop_x = len(x4_acc)-(speed3_end-speed3_start)
                y4_acc = y4_acc[:speed3_end-speed3_start]
                x4_acc = x4_acc[:speed3_end-speed3_start]
                total_results = total_results[:speed3_end-speed3_start]
            else:
                tail_crop_y = 0
                tail_crop_x = 0
                tail_crop = 0
            
            #==============================================================================
            # drawing Overall results
            #==============================================================================
            
            plt.figure(figsize=(12,6))
            plt.title(f"{title_plot} {algorithm}")
            if start_good_index:
                if good_index != 0:
                    plt.plot([good_index + i for i in range(len(total_results))], total_results, label="frequency model pred")
                    plt.axvline(good_index, 0, 200, color="brown", linestyle='dotted', linewidth=2, label="rPPG MAE < 10.0")
            else:
                plt.plot([speed3_start+i for i in range(len(total_results))], total_results, label="frequency model pred")
            plt.plot(y, label="ground truth")
            plt.plot([i+180 for i in range(len(x))], x, label="rppg pred")
            temp_value = speeds[coords_folders[input_num].split('.')[0]]
            plt.text(temp_value[0][1]-200, 70, 0, ha='center', va='bottom', fontsize=15, color='red')
            for value in temp_value:
                plt.axvline(value[1], 0, 200, color='red', linestyle='--', linewidth=2)
                plt.text(value[1]+80, 70, value[0], ha='center', va='bottom', fontsize=20, color='red')
            # plt.plot([i*interval_freq for i in range(len(ampl_input_list))], ampl_input_list, label="STFT")
            # plt.plot(coords_input_y_detrend, label="input data")
            plt.legend()
            plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_performance.png")
            plt.show()

            #==========================================================================================
            # Caculate and save My algorithm accuracy
            #==========================================================================================
            
            total_results_acc_mae = mean_absolute_error(y4_acc, total_results)
            total_results_acc_rmse = np.sqrt(mean_squared_error(y4_acc, total_results))
            total_results_acc_r2 = r2_score(y4_acc, total_results)
            total_results_acc_r2_adjust = adjusted_r2(total_results_acc_r2, len(y4_acc), len(total_results))
            dict4pearson = {
                "Frequency model":total_results,
                "Label":y4_acc}
            df4pearson = pd.DataFrame(dict4pearson)
            corr_matrix = df4pearson.corr(method='pearson')
            
            with open(f"./materials/fft_version_6_sec/{algorithm}/performance.txt", 'a') as f:
                f.write(f"{coords_folders[input_num]} freq:mae({total_results_acc_mae})rmse({total_results_acc_rmse})r2({total_results_acc_r2})\n")

            final_table_mae.loc['total', f'Frequency_{algorithm}'] += total_results_acc_mae
            final_table_rmse.loc['total', f'Frequency_{algorithm}'] += total_results_acc_rmse
            if -100 <= total_results_acc_r2 <= 100:
                final_table_r2.loc['total', f'Frequency_{algorithm}'] += total_results_acc_r2
                final_table_count4r2.loc['total', f'Frequency_{algorithm}'] += 1
            final_table_count.loc['total', f'Frequency_{algorithm}'] += 1
            final_table_pearson.loc['total', f'Frequency_{algorithm}'] += corr_matrix.loc["Frequency model", "Label"]
            
            if -100 <= total_results_acc_r2_adjust <= 100:
                final_table_r2_adjust.loc['total', f'Frequency_{algorithm}'] += total_results_acc_r2_adjust
                final_table_count4r2_adjust.loc['total', f'Frequency_{algorithm}'] += 1

            previous_point = temp_value[0][1]
            
            indices4partial_acc = []
            if start_good_index:
                for value in temp_value:
                    if value[1] <= good_index:
                        pass
                    else:
                        indices4partial_acc.append([value[0], value[1]-good_index])
            else:
                for value in temp_value[1:]:
                    indices4partial_acc.append([value[0], value[1]-speed3_start])
            
            previous_point = 0
            for index in range(len(indices4partial_acc)+1):
                try:
                    temp_freq = total_results[previous_point:indices4partial_acc[index][1]]
                    temp_y = y4_acc[previous_point:indices4partial_acc[index][1]]
                    previous_point = indices4partial_acc[index][1]
                except:
                    if isinstance(speed3_end, int):
                        break
                    else:
                        temp_freq = total_results[previous_point:]
                        temp_y = y4_acc[previous_point:]
                        index -= 1

                total_results_acc_mae = mean_absolute_error(temp_y, temp_freq)
                total_results_acc_rmse = np.sqrt(mean_squared_error(temp_y, temp_freq))
                total_results_acc_r2 = r2_score(temp_y, temp_freq)
                total_results_acc_r2_adjust = adjusted_r2(total_results_acc_r2, len(temp_y), len(temp_freq))
                dict4pearson = {
                    "Frequency model":temp_freq,
                    "Label":temp_y}
                df4pearson = pd.DataFrame(dict4pearson)
                corr_matrix = df4pearson.corr(method='pearson')
                
                # save accuracy
                
                with open(f"./materials/fft_version_6_sec/{algorithm}/performance.txt", 'a') as f:
                    f.write(f"{coords_folders[input_num]}speed({value[0]}) freq:mae({total_results_acc_mae})rmse({total_results_acc_rmse})r2({total_results_acc_r2})\n")
                
                final_table_mae.loc[int(temp_value[index][0]), f'Frequency_{algorithm}'] += total_results_acc_mae
                final_table_rmse.loc[int(temp_value[index][0]), f'Frequency_{algorithm}'] += total_results_acc_rmse
                if 0 <= total_results_acc_r2 <= 1:
                    final_table_r2.loc[int(temp_value[index][0]), algorithm] += total_results_acc_r2
                    final_table_count4r2.loc[int(temp_value[index][0]), algorithm] += 1
                final_table_count.loc[int(temp_value[index][0]), f'Frequency_{algorithm}'] += 1
                final_table_pearson.loc[int(temp_value[index][0]), f'Frequency_{algorithm}'] += corr_matrix.loc["Frequency model", "Label"]
                
                if 0 <= total_results_acc_r2_adjust <= 1:
                    final_table_r2_adjust.loc[int(temp_value[index][0]), f'Frequency_{algorithm}'] += total_results_acc_r2_adjust
                    final_table_count4r2_adjust.loc[int(temp_value[index][0]), f'Frequency_{algorithm}'] += 1

            #==========================================================================================
            # Caculate and save rPPG accuracy
            #==========================================================================================
            x_acc_mae = mean_absolute_error(y4_acc, x4_acc)
            x_acc_rmse = np.sqrt(mean_squared_error(y4_acc, x4_acc))
            x_acc_r2 = r2_score(y4_acc, x4_acc)
            x_acc_r2_adjust = adjusted_r2(x_acc_r2, len(y4_acc), len(x4_acc))
            dict4pearson = {
                "rPPG model":x4_acc,
                "Label":y4_acc}
            df4pearson = pd.DataFrame(dict4pearson)
            corr_matrix = df4pearson.corr(method='pearson')
            
            with open(f"./materials/fft_version_6_sec/{algorithm}/performance.txt", 'a') as f:
                f.write(f"{coords_folders[input_num]} x:mae({x_acc_mae})rmse({x_acc_rmse})r2({x_acc_r2})\n")
            
            final_table_mae.loc['total', algorithm] += x_acc_mae
            final_table_rmse.loc['total', algorithm] += x_acc_rmse
            if -100 <= x_acc_r2 <= 100:
                final_table_r2.loc['total', algorithm] += x_acc_r2
                final_table_count4r2.loc['total', algorithm] += 1
            final_table_count.loc['total', algorithm] += 1
            final_table_pearson.loc['total', algorithm] += corr_matrix.loc["rPPG model", "Label"]
            
            if -100 <= x_acc_r2_adjust <= 100:
                final_table_r2_adjust.loc['total', algorithm] += x_acc_r2_adjust
                final_table_count4r2_adjust.loc['total', algorithm] += 1
            
            previous_point = 0
            for index in range(len(indices4partial_acc)+1):
                try:
                    temp_x = x4_acc[previous_point:indices4partial_acc[index][1]]
                    temp_y = y4_acc[previous_point:indices4partial_acc[index][1]]
                    previous_point = indices4partial_acc[index][1]
                except:
                    if isinstance(speed3_end, int):
                        break
                    else:
                        temp_x = x4_acc[previous_point:]
                        temp_y = y4_acc[previous_point:]
                        index -= 1
                
                x_acc_mae = mean_absolute_error(temp_y, temp_x)
                x_acc_rmse = np.sqrt(mean_squared_error(temp_y, temp_x))
                x_acc_r2 = r2_score(temp_y, temp_x)
                x_acc_r2_adjust = adjusted_r2(x_acc_r2, len(temp_y), len(temp_x))
                total_results_acc_r2_adjust = adjusted_r2(total_results_acc_r2, len(temp_y), len(temp_freq))
                dict4pearson = {
                    "rPPG model":temp_x,
                    "Label":temp_y}
                df4pearson = pd.DataFrame(dict4pearson)
                corr_matrix = df4pearson.corr(method='pearson')
                
                # save accuracy
                
                with open(f"./materials/fft_version_6_sec/{algorithm}/performance.txt", 'a') as f:
                    f.write(f"{coords_folders[input_num]}speed({value[0]}) x:mae({x_acc_mae})rmse({x_acc_rmse})r2({x_acc_r2})\n")
                
                final_table_mae.loc[int(temp_value[index][0]), algorithm] += x_acc_mae
                final_table_rmse.loc[int(temp_value[index][0]), algorithm] += x_acc_rmse
                if 0 <= x_acc_r2 <= 1:
                    final_table_r2.loc[int(temp_value[index][0]), algorithm] += x_acc_r2
                    final_table_count4r2.loc[int(temp_value[index][0]), algorithm] += 1
                final_table_count.loc[int(temp_value[index][0]), algorithm] += 1
                final_table_pearson.loc[int(temp_value[index][0]), algorithm] += corr_matrix.loc["rPPG model", "Label"]
                
                if 0 <= x_acc_r2_adjust <= 1:
                    final_table_r2_adjust.loc[int(temp_value[index][0]), algorithm] += x_acc_r2_adjust
                    final_table_count4r2_adjust.loc[int(temp_value[index][0]), algorithm] += 1
            

            #==========================================================================================
            # Drawing input data and inter-process values for my algorithm
            #==========================================================================================
            diff_final_integral = []
            for i, value in enumerate(diff_final_log):
                if i == 0:
                    diff_final_integral.append(value)
                else:
                    diff_final_integral.append(diff_final_integral[i-1]+value)
            
            plt.title(f"{title_plot} Input data")
            plt.plot(minmax(acc_log), label="acc", zorder=0)
            plt.plot(acc_sign_log, label="acc_diff_1", zorder=1)
            plt.plot(diff_acc_sign_coefficient_log, label="acc_diff_1_mv", zorder=2)
            plt.plot(diff_final_log, label="hr variance", zorder=3)
            # plt.plot(minmax(diff_final_integral), label="diff final integral", zorder=4)
            plt.plot(np.zeros(len(acc_log)), c="black", zorder=4)
            plt.plot(minmax(diff_main_log), label="diff_main", zorder=5)
            plt.plot()
            plt.legend()
            plt.savefig(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_input_data.png")
            plt.show()
            
            np.save(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_y.npy", np.array(y))
            np.save(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_x.npy", np.array(x))
            np.save(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_total_results.npy", np.array(total_results))
            np.save(f"./materials/fft_version_6_sec/{algorithm}/{coords_folders[input_num]}_ampl_input_list.npy", np.array(ampl_input_list))
    
    #==========================================================================================
    # Save accuracy
    #==========================================================================================
    
    final_table_mae = final_table_mae/final_table_count
    final_table_rmse = final_table_rmse/final_table_count
    final_table_r2 = final_table_r2/final_table_count4r2
    final_table_r2_adjust = final_table_r2_adjust/final_table_count4r2_adjust
    final_table_pearson = final_table_pearson/final_table_count
    final_table_spectrogram_pearson = final_table_spectrogram_pearson/final_table_spectrogram_pearson_count
    
    final_table_mae.to_excel(f"./materials/fft_version_6_sec/mae_table.xlsx")
    final_table_rmse.to_excel(f"./materials/fft_version_6_sec/rmse_table.xlsx")
    final_table_r2.to_excel(f"./materials/fft_version_6_sec/r2_table.xlsx")
    final_table_r2_adjust.to_excel(f"./materials/fft_version_6_sec/r2_adjust_table.xlsx")
    final_table_pearson.to_excel(f"./materials/fft_version_6_sec/pearson_table.xlsx")
    final_table_spectrogram_pearson.to_excel(f"./materials/fft_version_6_sec/spectrogram_pearson_table.xlsx")
    
    #==========================================================================================
    # Drawing accuracy
    #==========================================================================================
    
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(final_table_mae, annot=True, vmin=10.0, vmax=100.0)
    plt.title("total mae")
    plt.savefig(f"./materials/fft_version_6_sec/mae_table.png")
    plt.show()
    
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(final_table_rmse, annot=True, vmin=10.0, vmax=100.0)
    plt.title("total rmse")
    plt.savefig(f"./materials/fft_version_6_sec/rmse_table.png")
    plt.show()
    
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(final_table_r2, annot=True, vmin=0.0, vmax=1.0)
    plt.title("total r2")
    plt.savefig(f"./materials/fft_version_6_sec/r2_table.png")
    plt.show()
    
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(final_table_r2_adjust, annot=True, vmin=0.0, vmax=1.0)
    plt.title("total r2")
    plt.savefig(f"./materials/fft_version_6_sec/r2_adjust_table.png")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(final_table_pearson, annot=True, vmin=-1.0, vmax=1.0, fmt=".2f")
    annotations = final_table_pearson.applymap(lambda x: 'NaN' if pd.isna(x) else f"{x:.1f}")
    mask = final_table_pearson.isna()
    sns.heatmap(final_table_pearson, mask=~mask, annot=annotations, fmt=".2f", cmap='Pastel1',
            cbar=False, linewidths=.5, linecolor='gray',
            annot_kws={"fontsize": 12, "color": "red"})
    plt.title("total pearson")
    plt.savefig(f"./materials/fft_version_6_sec/pearson_table.png")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(final_table_spectrogram_pearson, annot=True, vmin=-1.0, vmax=1.0, fmt=".2f")
    plt.title("total pearson")
    plt.savefig(f"./materials/fft_version_6_sec/spectrogram_pearson_table.png")
    plt.show()
