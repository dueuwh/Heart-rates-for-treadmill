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
from frequency_model import fft_half
from params import Params
from copy import deepcopy
import queue

# for BO
import optuna

def HPO_process(hyperparameters, model):
    
    # === hyperparameter optimization process === #
    
    return model

def freq_model_pipeline(ldmk_list, stft_window):
    
    # === frequency_model_setup === #
    
    model_1 = freq_model(Params.frequency_model_hyperparameters)
    
    # === skin extraction === #
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    
    frame_dir = os.getcwd()+"/data/frames/"
    dataset_list = os.listdir(frame_dir)
    
    sel_dataset = dataset_list[0]
    sel_dataset = "KDY_2"
    frame_list = os.listdir(frame_dir + sel_dataset)
    
    frame_list = sorted(frame_list)
    
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
    
    for frame in frame_list:
        load_frame = np.load(frame_dir+sel_dataset+f'/{frame}')
        
        image = cv2.cvtColor(load_frame, cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.show()
        
        processed_frames_count += 1
    
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

def run_optuna_OP():
    study = optuna.create_study(direction="minimize")
    study.optimize(freq_model_pipeline, n_trials=1000)
    
    best_params = study.best_value
    loss_of_best_params = study.best_value
    print(f"loss of best paprams: {loss_of_best_params}")

if __name__ == "__main__":
    landmark_list = [4]
    stft_window = 180  # 30(fps) * 6(seconds) 6 seconds is the time window used in pyVHR pipeline
    freq_model_pipeline(landmark_list, stft_window)