# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:18:53 2024

@author: ys
"""

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
import cv2
from pyVHR.extraction.skin_extraction_methods import SkinExtractionConvexHull
from copy import deepcopy
import queue
import seaborn as sns
from utils import *

def coord_preprocessing(coords_series):
    num_row, num_col = coords_series[0].shape
    
    x_series = []
    y_series = []
    
    for coord_frame in coords_series:
        temp_summation = np.sum(coord_frame, axis=0)
        x_series.append(temp_summation[0])
        y_series.append(temp_summation[1])
    return x_series, y_series

def rppg_pipeline(ldmk_list, data_folder, stride, window):
    
    # === skin extraction === #
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    
    frame_dir = os.getcwd()+ "/data/frames/" + data_folder + '/'
    frame_list = os.listdir(frame_dir)
    
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
    
    
    sig_list_max = int(fps*window)
    passed_sig_max = int(fps*stride)
    
    sig_list = []
    passed_sig = 0
    
    for frame in frame_list[5000:]:
        load_frame = np.load(frame_dir+frame)
        
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
                cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, landmark_coords)
                frame_list.append(cropped_skin_im)
                
        else:
            print(f"A face is not detected {processed_frames_count}")
            cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, landmark_coords)  # If face detection is failed, cropping same coordinates
            frame_list.append(cropped_skin_im)
            full_skin_im = image
        
        # landmark check
        # plt.imshow(image)
        # for i in range(coord_arr.shape[0]):`
        #     plt.scatter(coord_arr[i, 0], coord_arr[i, 1], s=1, c='r')
        # plt.show()
        
        # cropped_skin_im check
        # plt.imshow(cropped_skin_im)
        # plt.title(f"cropped_skin_im {processed_frames_count}")
        # plt.show()
        
        sig_list.append(holistic_mean(cropped_skin_im, np.int32(55), np.int32(200))[0, :])
        
        
        if len(sig_list) > sig_list_max:
            if passed_sig > passed_sig_max:
                passed_sig = 0
                del sig_list[0]
                # pre-filtering
                filter_input = np.array(sig_list)
                sig_arr = signal_filtering(filter_input)
                
                # bvp
                # sig_list = np.expand_dims(sig_list, axis=0)
                bvp_arr = cpu_OMIT(sig_arr.T)
                print("bvp_arr.shape: ", bvp_arr.shape)
                
                # post-filtering
                bvp_arr = signal_filtering(bvp_arr)
                
                pred_welch_hr, SNR, pSNR, Pfreqs, Power = BPM(data=bvp_arr, fps=fps, startTime=0, minHz=0.5, maxHz=4.0, verb=False).BVP_to_BPM()
                
                if SNR > SNR_threshold:
                    print("bpm: ", pred_welch_hr)
                else:
                    print("bpm is invalid (SNR)")
            else:
                del frame_list[0]
                passed_sig += 1
            
            
        processed_frames_count += 1
        if processed_frames_count%30 == 0:
            print(f"{round((processed_frames_count/total_frame)*100, 2)}%")

if __name__ == "__main__":
    fps = 30
    frame_dir = os.getcwd()+"/data/frames/"
    folder_list = os.listdir(frame_dir)
    select_data = folder_list[0]
    landmark_list = [i for i in range(468)]
    stride = 1  # seconds
    window = 6  # seconds
    rppg_pipeline(landmark_list, select_data, stride, window)
    
# =============================================================================
#     pure_dataset = "D:/home/rPPG/data/PURE_rPPG/"
#     p_loader = pure_loader(pure_dataset)
#     pure_signal = p_loader.readSigfile(pure_dataset+'01-01.json')
#     plt.plot(pure_signal)
#     plt.title("pure_signal 01-01.json")
#     plt.xlabel("Time")
#     plt.ylabel("rPPG wave amplitude")
#     plt.show()
# =============================================================================
        