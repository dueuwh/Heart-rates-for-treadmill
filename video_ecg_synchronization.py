import os
import sys
sys.executable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2

base_path = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/"
frame_path = f"{base_path}frames/"
frame_list = os.listdir(frame_path)
label_path = f"{base_path}labels/"
video_path = f"{base_path}videos/"
video_list = os.listdir(video_path)
save_path = f"{base_path}labels_synchro/"

# for video_name in video_list:
#     if video_name.split('.')[0] not in frame_list:
#         print(f"Frame folder {video_name} is not exist.\
#               Start frame extraction.")
#         cap = cv2.VideoCapture(f"{video_path}{video_name}")
#         save_path = f"{frame_path}{video_name.split('.')[0]}"
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
        
#         frame_count = 10000000
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             cv2.imwrite(f"{save_path}/{frame_count}.jpg", frame)
#             frame_count += 1
#         print(f"{video_name} frame extraction is finished.")

# print("All done")

for frame_folder in frame_list:
    frame_length = len(os.listdir(f"{frame_path}{frame_folder}"))
    
    label_name = frame_folder[:5]
    label_length = pd.read_csv(f"{label_path}{label_name}.csv")["ECG"].dropna().to_numpy().shape[0]
    hr_length = pd.read_csv(f"{label_path}{label_name}.csv")["HR"].dropna().to_numpy().shape[0]
    label = pd.read_csv(f"{label_path}{label_name}.csv")["ECG"].dropna().to_numpy()
    original_indices = np.arange(label_length)
    new_indices = np.linspace(0, label_length-1, frame_length)
    interp_label = np.interp(new_indices, original_indices, label)
    
    label_save_path = f"{save_path}{label_name}.npy"
    
    print(f"\nfolder: {frame_folder}, frame_length: {frame_length}\
          label_length: {label_length}, HR_length: {hr_length}\
              interpolated label_length: {len(interp_label)}")
    
    if frame_length == len(interp_label):
        np.save(label_save_path, interp_label)
    else:
        raise ValueError(f"Interpolated label length: {len(interp_label)} is not same as frame_length: {frame_length}")