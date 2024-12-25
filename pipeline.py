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
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

class UKFFilter:
    def __init__(self, dt=1.0, 
                 initial_state=np.array([0, 0, 1, 0.5]),
                 initial_covariance=np.eye(4) * 500.,
                 process_noise_std=0.1,
                 measurement_noise_std=2.0):
        """
        UKF 초기화
        """
        self.dt = dt
        
        # 상태 전이 함수 정의
        def fx(x, dt):
            F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1,  0],
                          [0, 0, 0,  1]])
            return F.dot(x)
        
        # 관측 함수 정의
        def hx(x):
            return np.array([x[0], x[1]])
        
        # Sigma 포인트 초기화
        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=1.)
        
        # UKF 객체 생성
        self.ukf = UKF(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=self.dt, points=points)
        
        # 초기 상태 설정
        self.ukf.x = initial_state
        
        # 초기 공분산 행렬 설정
        self.ukf.P = initial_covariance
        
        # 프로세스 노이즈 공분산 행렬 설정
        self.ukf.Q = np.eye(4) * process_noise_std**2
        
        # 측정 노이즈 공분산 행렬 설정
        self.ukf.R = np.eye(2) * measurement_noise_std**2
    
    def process_measurement(self, measurement):
        """
        측정값을 받아 UKF 업데이트 수행하고 추정된 상태를 반환
        
        Parameters:
        - measurement (tuple or list or np.ndarray): (x, y) 측정값
        
        Returns:
        - estimated_state (np.ndarray): 추정된 상태 벡터 [x, y, vx, vy]
        """
        measurement = np.array(measurement)
        self.ukf.predict()
        self.ukf.update(measurement)
        return self.ukf.x.copy()
    
    def predict_only(self):
        """
        UKF 예측 단계만 수행하고 예측된 상태를 반환
        
        Returns:
        - predicted_state (np.ndarray): 예측된 상태 벡터 [x, y, vx, vy]
        """
        self.ukf.predict()
        return self.ukf.x.copy()

def coord_preprocessing(coords_series):
    num_row, num_col = coords_series[0].shape
    
    x_series = []
    y_series = []
    
    for coord_frame in coords_series:
        temp_summation = np.sum(coord_frame, axis=0)
        x_series.append(temp_summation[0])
        y_series.append(temp_summation[1])
    return x_series, y_series

def rppg_pipeline(ldmk_list, data_folder, stride, window, save_dir):
    
    # === skin extraction === #
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    
    frame_dir = os.getcwd()+ "/data/treadmill_dataset/frames/" + data_folder + '/'
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
    
    coord_list = []
    
    ukf_filter = UKFFilter(dt=1.0, 
                           initial_state=np.array([0, 0, 1, 0.5]),
                           initial_covariance=np.eye(4) * 500.,
                           process_noise_std=0.1,
                           measurement_noise_std=2.0)
    
    for frame in frame_list:
        load_frame = cv2.imread(frame_dir+frame)
        
        image = cv2.cvtColor(load_frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image)
        
        if results.multi_face_landmarks:
            # for face_landmarks in results.multi_face_landmarks:
            #     ldmk_counter = 0
            #     landmark_coords = np.zeros((468, 2), dtype=np.float32)
            #     coord_row_counter = 0
            #     for lm in face_landmarks.landmark:
            #         coords = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, width, height)
            #         print("coords: ", coords)
            #         landmark_coords[ldmk_counter, 0] = coords[1]
            #         landmark_coords[ldmk_counter, 1] = coords[0]
            #         if ldmk_counter in ldmk_list:
            #             coord_arr[coord_row_counter, :] = np.array(coords)
            #             coord_row_counter += 1
            #         ldmk_counter += 1
            #     cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, landmark_coords)
            #     # frame_list.append(cropped_skin_im)
            current_position = (results.multi_face_landmarks[0].landmark[4].x, results.multi_face_landmarks[0].landmark[4].y)
            # filtered_state = ukf_filter.process_measurement(current_position)
            # current_position = (filtered_state[0]*image.shape[1], filtered_state[1]*image.shape[0])
        else:
            print(f"A face is not detected {processed_frames_count}")
            # cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, landmark_coords)  # If face detection is failed, cropping same coordinates
            # frame_list.append(cropped_skin_im)
            # full_skin_im = image
            
            # if len(coord_list) > 0:
            #     predicted_state = ukf_filter.predict_only()
            #     current_position = (predicted_state[0]*image.shape[1], predicted_state[1]*image.shape[0])
            # else:
            #     print(f"\n\nfirst position is not detected!!!\n\n")
            #     current_position = (ukf_filter.ukf.x[0]*image.shape[1], ukf_filter.ukf.x[1]*image.shape[0])
            current_position = (0,0)
            
        coord_list.append(current_position)
        
        processed_frames_count += 1
        if processed_frames_count%30 == 0:
            print(f"{round((processed_frames_count/total_frame)*100, 2)}%")
            
    np.save(f"{save_dir}{data_folder}.npy", np.array(coord_list))
    plt.title(data_folder)
    plt.plot(coord_list)
    plt.show()
        
        # landmark check
        # plt.imshow(image)
        # for i in range(coord_arr.shape[0]):`
        #     plt.scatter(coord_arr[i, 0], coord_arr[i, 1], s=1, c='r')
        # plt.show()
        
        # cropped_skin_im check
        # plt.imshow(cropped_skin_im)
        # plt.title(f"cropped_skin_im {processed_frames_count}")
        # plt.show()
        
        # sig_list.append(holistic_mean(cropped_skin_im, np.int32(55), np.int32(200))[0, :])
        
        
        # if len(sig_list) > sig_list_max:
        #     if passed_sig > passed_sig_max:
        #         passed_sig = 0
        #         del sig_list[0]
        #         # pre-filtering
        #         filter_input = np.array(sig_list)
        #         sig_arr = signal_filtering(filter_input)
                
        #         # bvp
        #         # sig_list = np.expand_dims(sig_list, axis=0)
        #         bvp_arr = cpu_OMIT(sig_arr.T)
        #         print("bvp_arr.shape: ", bvp_arr.shape)
                
        #         # post-filtering
        #         bvp_arr = signal_filtering(bvp_arr)
                
        #         pred_welch_hr, SNR, pSNR, Pfreqs, Power = BPM(data=bvp_arr, fps=fps, startTime=0, minHz=0.5, maxHz=4.0, verb=False).BVP_to_BPM()
                
        #         if SNR > SNR_threshold:
        #             print("bpm: ", pred_welch_hr)
        #         else:
        #             print("bpm is invalid (SNR)")
        #     else:
        #         del frame_list[0]
        #         passed_sig += 1
            

if __name__ == "__main__":
    fps = 30
    frame_dir = os.getcwd()+"/data/treadmill_dataset/frames/"
    frame_folders = os.listdir(frame_dir)
    
    
    save_dir = os.getcwd()+"/data/coordinates/raw/"
    saved_files = [name.split('.')[0] for name in os.listdir(save_dir)]
    # frame_folders = [name for name in frame_folders if name not in saved_files]
    frame_folders = frame_folders[10:]
    landmark_list = [i for i in range(468)]
    stride = 1  # seconds
    window = 6  # seconds
    for select_data in frame_folders:
        print(select_data)
        rppg_pipeline(landmark_list, select_data, stride, window, save_dir)
    
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
        