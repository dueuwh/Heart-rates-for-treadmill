# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:00:48 2024

@author: ys
"""

# commit test

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
        
        self.sig_coeffi = hyperparameter_setting["sigmoid_power_coefficient"]
        self.sig_const = hyperparameter_setting["sigmoid_power_constant"]
        self.sig_numer = hyperparameter_setting["sigmoid_numerator"]
        
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
            
            # Utilizing an exponential function to progressively reduce the rate of change in BPM over frequency
            bpm_output = self.bpm_previous + diff_final * self.scaling * self.exp_base_scaling ** (self.exp_power_scaling * abs(diff_final)) + self.bias
            
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


if __name__ == "__main__":
    model = freq_model(Params.frequency_model_hyperparameters)
    input_base_dir = "D:/home/BCML/drax/PAPER/data/coordinates/stationary_kalman/"
    label_base_dir = "D:/home/BCML/drax/PAPER/data/labels/synchro/"
    file_list = os.listdir(input_base_dir)
    label_list = os.listdir(label_base_dir)
    # label_list = [name for name in os.listdir(label_base_dir) if 'synchro' in name]
    
    fft_window = 300
    
    pass_band = (0.5, 10)
    biquad_section = 10
    sampling_rate = 30  # video FPS
    sos = signal.butter(biquad_section, pass_band, 'bandpass', fs=sampling_rate, output='sos')
    
    total_results = {}
    
    for file in file_list:
        
        if "LHT_1" in file:
            continue
        
        for name in label_list:
            if file.split('.')[0] in name:
                load_label_name = name
                break
        
        label = pd.read_csv(label_base_dir+load_label_name, index_col=0)
        try:
            label = label["HR"].tolist()
        except KeyError:
            label = label['0'].tolist()
            
        load = np.load(input_base_dir+file)
        
        x = load[4, 0, :]
        y = load[4, 1, :]
        
        total_results[file] = []
        
        est_axis = []
        none_list = []
        none_axis = []
        
        print(f"{file}: {len(label)}")
        for i in range(y.shape[0]-fft_window+1):
            ldmk_bpf = signal.sosfilt(sos, y[i:i+fft_window])
            
            ampl_input, _ = fft_half(ldmk_bpf, sampling_rate, fft_window)
            
            if i == 0:
                run_start = True
                initial_bpm = label[fft_window] + np.random.randint(1, 6)
                est_bpm = model.run(ampl_input, run_start, initial_bpm)
            else:
                run_start = False
                est_bpm = model.run(ampl_input, run_start)
            
            if est_bpm is not None:
                total_results[file].append(est_bpm)
                est_axis.append(i+fft_window)
            else:
                none_list.append(1)
                none_axis.append(i)
        
        print(f"label lenght: {len(label)}, est_axis max: {max(est_axis)}")
        if len(label) < len(est_axis):
            est_axis = ext_axis[:-abs(len(label) - len(est_axis))]
            total_results[file] = total_results[file][:-abs(len(label) - len(est_axis))]
            
        label_loss = [label[i-2] for i in est_axis]
        mae = round(mean_absolute_error(label_loss, total_results[file]), 2)
        mse = round(mean_squared_error(label_loss, total_results[file]), 2)
        rmse = round(np.sqrt(mse), 2)
        r2 = round(r2_score(label_loss, total_results[file]), 2)
        
        plt.scatter(est_axis, total_results[file], label="pred", s=1, c='r')
        # plt.scatter(none_axis, none_list, label="none", s=5, c='g')
        plt.plot(label, label="label")
        plt.title(f"{file} mae: {mae}, mse: {mse}, rmse: {rmse}, r square: {r2}")
        plt.legend()
        plt.xlabel("frame number")
        plt.ylabel("BPM")
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

# ref_list = [name.split('.')[0]+'.csv' for name in video_list]
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