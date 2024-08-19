# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:17:33 2024

@author: ys
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mediapipe as mp
import librosa
import librosa.display
import cv2
from pyVHR.extraction.skin_extraction_methods import SkinExtractionConvexHull
from frequency_model import non_DNN_model as freq_model
from frequency_model import fft_half, fft_half_normalization_test
from params import Params
from copy import deepcopy
import queue
import seaborn as sns
import librosa
from ukf import UKF
from utils import *

import optuna

def HPO_process(hyperparameters, model):
    
    # === hyperparameter optimization process === #
    
    return model

def coord_preprocessing(coords_series):
    num_row, num_col = coords_series[0].shape
    
    x_series = []
    y_series = []
    
    for coord_frame in coords_series:
        temp_summation = np.sum(coord_frame, axis=0)
        x_series.append(temp_summation[0])
        y_series.append(temp_summation[1])
    return x_series, y_series

def freq_model_pipeline(ldmk_list, stft_window):
    
    # === frequency_model_setup === #
    
    model_1 = freq_model(Params.frequency_model_hyperparameters)
    
    # === skin extraction === #
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    
    frame_dir = os.getcwd()+"/data/frames/"
    dataset_list = os.listdir(frame_dir)
    
    for sel_dataset in dataset_list:
        frame_list = os.listdir(frame_dir + sel_dataset)
        
        frame_list = sorted(frame_list)
        total_frame = len(frame_list)
        
        # face_detector = os.getcwd()+"/face_detection_opencv/"
        # detector = cv2.dnn.readNetFromCaffe(f"{face_detector}/deploy.prototxt" , f"{face_detector}res10_300x300_ssd_iter_140000.caffemodel")
        
        fps = 30.0
        frame_timestamp = 0
        
        width = 640
        height = 480
        
        skin_ex = SkinExtractionConvexHull('CPU')
        
        processed_frames_count = 0
        
        coord_row = len(ldmk_list)
        coord_col = 2
        # coord_ch = len(frame_list)
        coord_arr = np.zeros((coord_row, coord_col))
        # print("coordinate array shape: ", coord_arr.shape)
        
        coord_list = []
        coord_list_librosa = []
        
        freq_list_x = []
        freq_list_y = []
        
        freq_temp_arr_x = np.zeros((int(fps), int(stft_window/2)))
        freq_temp_arr_y = np.zeros((int(fps), int(stft_window/2)))
        freq_temp_arr_counter = 0
        for frame in frame_list:
            load_frame = np.load(frame_dir+sel_dataset+f'/{frame}')
            
            image = cv2.cvtColor(load_frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(image)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ldmk_counter = 0
                    landmark_coords = np.zeros((468, 2), dtype=np.float32)
                    coord_row_counter = 0
                    for lm in face_landmarks.landmark:
                        coords = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, width, height)
                    
                        landmark_coords[ldmk_counter, 0] = coords[1]
                        landmark_coords[ldmk_counter, 1] = coords[0]
                        if ldmk_counter in ldmk_list:
                            coord_arr[coord_row_counter, :] = np.array(coords)
                            coord_row_counter += 1
                        ldmk_counter += 1
                        if ldmk_counter > ldmk_list[-1]:
                            coord_list.append(coord_arr)
                            coord_arr = np.zeros((coord_row, coord_col))
                            break
                coord_list_librosa.append(coord_arr)
                    
                    # for rPPG-based algorithm
                    # cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, landmark_coords)
                    
            else:
                print(f"A face is not detected {processed_frames_count}")
                cropped_skin_im = np.zeros_like(image)
                full_skin_im = np.zeros_like(image)
            
            # landmark check
            # plt.imshow(image)
            # plt.scatter(coord_arr[0, 0], coord_arr[0, 1], s=2, c='r')  # draw landmark on raw frame
            # plt.show()
            
            if len(coord_list) >=stft_window:
                del coord_list[0]
                x_s, y_s = coord_preprocessing(coord_list)
                # print(f"coord_list:\n{coord_list}\nx_s:\n{x_s}\ny_s:\n{y_s}\n========================\n")
                freq_temp_arr_x[freq_temp_arr_counter]=fft_half(x_s, fps, stft_window)[0]  # scaling?
                freq_temp_arr_y[freq_temp_arr_counter]=fft_half(y_s, fps, stft_window)[0]  # scaling?
                freq_temp_arr_counter += 1
            
                if processed_frames_count % fps == 0 and processed_frames_count != 0:
                    freq_temp_arr_counter = 0
                    freq_list_x.append(np.sum(freq_temp_arr_x, axis=0))
                    freq_list_y.append(np.sum(freq_temp_arr_x, axis=0))
            
            processed_frames_count += 1
            if processed_frames_count%30 == 0:
                print(f"{sel_dataset} : {round((processed_frames_count/total_frame)*100, 2)}%")
        
        x_s, y_s = coord_preprocessing(coord_list_librosa)
        
        plt.plot(x_s, label="x_coordinates")
        plt.plot(y_s, label="y_coordinates")
        plt.legend()
        plt.title("coordinates")
        plt.xlabel("frames")
        plt.ylabel("value")
        plt.show()
        
        if sel_dataset not in os.listdir(os.getcwd()+'/data/coordinates/'):
            os.mkdir(os.getcwd()+'/data/coordinates/'+sel_dataset)
        coordinates_save_path = os.getcwd()+'/data/coordinates/'+sel_dataset+'/'
        
        with open(f"{coordinates_save_path}x.txt", 'w') as f_x:
            for value in x_s:
                f_x.write(str(value)+' ')
        
        with open(f"{coordinates_save_path}y.txt", 'w') as f_y:
            for value in y_s:    
                f_y.write(str(value)+' ')
            
        
        x_librosa_fft = librosa.stft(np.array(x_s), win_length=stft_window, hop_length=int(stft_window/2))
        x_spectrogram = np.abs(x_librosa_fft)**2
        y_librosa_fft = librosa.stft(np.array(y_s), win_length=stft_window, hop_length=int(stft_window/2))
        y_spectrogram = np.abs(y_librosa_fft)**2
        
        spectrogram_save_path = f'./data/spectrogram/{sel_dataset}/'
        if sel_dataset not in os.listdir('./data/spectrogram/'):
            os.mkdir(spectrogram_save_path)
        
        duration = len(frame_list)/fps
        plt.figure(figsize=(15,10))
        plt.imshow(np.log10(x_spectrogram), aspect='auto', origin='lower', cmap='jet', extent=(0, duration, 0, fps/2))
        plt.xlabel("Time: [s]")
        plt.ylabel("Frequency: [Hz]")
        plt.title("Spectrogram X")
        plt.colorbar(label="Magnitude (dB)")
        plt.savefig(spectrogram_save_path+'X_plot.jpg')
        plt.show()
        
        plt.figure(figsize=(15,10))
        plt.imshow(np.log10(y_spectrogram), aspect='auto', origin='lower', cmap='jet', extent=(0, duration, 0, fps/2))
        plt.xlabel("Time: [s]")
        plt.ylabel("Frequency: [Hz]")
        plt.title("Spectrogram Y")
        plt.colorbar(label="Magnitude (dB)")
        plt.savefig(spectrogram_save_path+'Y_plot.jpg')
        plt.show()
        
        with open(f'./data/spectrogram/mid_save/{sel_dataset}_x.txt', 'w') as f_x:
            for arr in freq_list_x:
                temp = ""
                for value in arr:
                    temp += f"{str(value)} "
                f_x.write(temp+'\n')
        
        with open(f'./data/spectrogram/mid_save/{sel_dataset}_y.txt', 'w') as f_y:
            for arr in freq_list_y:
                temp = ""
                for value in arr:
                    temp += f"{str(value)} "
                f_y.write(temp+'\n')
        
        print("txt file saved")
        
        df_x = pd.DataFrame(columns=[i for i in range(int(stft_window/2))])
        df_y = pd.DataFrame(columns=[i for i in range(int(stft_window/2))])
        
        for i in range(len(freq_list_x)):
            df_x.loc[i] = freq_list_x[i]
            df_y.loc[i] = freq_list_y[i]
        
        df_x.to_csv(f"./data/spectrogram/mid_save/{sel_dataset}_x.csv", index=False)
        df_y.to_csv(f"./data/spectrogram/mid_save/{sel_dataset}_y.csv", index=False)
    
    #==========================================================================
    #
    # To Do (development)
    # 1. To check the spectrogram of frequency model                             
    # 2. Testing the algorithm with current hyperparameters                      
    # 3. To use BOHB or other HPO algorithm for frequency model                  
    # 4. Combining the pyvhr process (rPPG model) - only in first few seconds    
    # 5. Postprocessing - UKF?                                                   
    # 6. Additional preprocessing - face landmark trajectory smoothing           
    #
    # To Do (paper)
    # 1. To save spectrogram figures and table after completing dev list item 1  
    # 2. Pearson correlation, Correlation Coefficient, ACF, Distance metric...   
    # 3. Model results figures and table after completing dev list item 5        
    # and writing the paper                                                      
    #
    #==========================================================================

def freq_analysis(num_fft):
    base_dir = "./data/spectrogram/mid_save/"
    file_list = os.listdir(base_dir)
    
    df_x = pd.DataFrame(columns=[i for i in range(int(num_fft/2))])
    df_y = pd.DataFrame(columns=[i for i in range(int(num_fft/2))])
    line_counter = 0
    
    with open(base_dir+file_list[0], 'r') as f:
        for line in f:
            value_list= line.split(' ')
            value_list.remove('\n')
            temp_value_float_list = [float(value) for value in value_list]
            df_x.loc[line_counter] = temp_value_float_list
            line_counter += 1
    
    df_x = df_x.transpose()
    
    sns.heatmap(df_x, annot=False, cmap="YlGnBu")
    plt.xlabel("seconds")
    plt.ylabel("frequency")
    plt.title("x spectrogram")
    plt.show()
    
def run_optuna_OP():
    study = optuna.create_study(direction="minimize")
    study.optimize(freq_model_pipeline, n_trials=1000)
    
    best_params = study.best_value
    loss_of_best_params = study.best_value
    print(f"loss of best paprams: {loss_of_best_params}")

if __name__ == "__main__":
    landmark_list = [4]
    landmark_list = sorted(landmark_list)
    print("landmark_list: ", landmark_list)
    stft_window = 180  # 30(fps) * 6(seconds) 6 seconds is the time window used in pyVHR pipeline
    
    print("run function: ", end=' ')
    function = int(input())
    
    if function==1:
        print("run pipeline")
        freq_model_pipeline(landmark_list, stft_window)
    elif function==2:
        print("loading mid-save")
        freq_analysis(stft_window)