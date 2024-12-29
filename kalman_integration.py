# kalman_integration.py

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import natsort
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pandas as pd
import seaborn as sns

def load_data(base_path, folder):
    """
    지정된 폴더 내의 모든 데이터 파일을 로드하여 시간 벡터와 함께 반환합니다.
    
    Parameters:
    base_path (str): 데이터가 저장된 기본 경로
    folder (str): 특정 데이터 폴더 이름
    
    Yields:
    tuple: (t, true_state, sensor1, sensor2, title)
    """
    total_files = [name for name in os.listdir(f"{base_path}{folder}") if '.npy' in name]
    y_total = natsort.natsorted([name for name in total_files if '_y' in name])
    x_total = natsort.natsorted([name for name in total_files if '_x' in name])
    freq_total = natsort.natsorted([name for name in total_files if '_total_results' in name])
    index_total = natsort.natsorted([name for name in total_files if '_time_index' in name])
    
    for i in range(len(y_total)):
        true_state = np.load(f"{base_path}{folder}/{y_total[i]}")
        sensor1 = np.load(f"{base_path}{folder}/{freq_total[i]}")
        sensor2 = np.load(f"{base_path}{folder}/{x_total[i]}")
        index = []
        with open(f"{base_path}{folder}/{index_total[i]}", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                split_word = line.split('_')
                index.append([split_word[0], split_word[1]])
        
        n_steps = len(true_state)  # 데이터 길이
        dt = 1.0  # 실제 시간 간격으로 수정 (데이터에 맞게 조정)
        t = np.linspace(0, dt * (n_steps - 1), n_steps)  # 시간 벡터 생성
        
        
        yield t, true_state, sensor1, sensor2, y_total[i], index

def fx(x, dt):
    """
    상태 전이 함수: 상태가 선형적으로 변한다고 가정.
    여기서는 상태가 동일하게 유지된다고 가정합니다.
    
    Parameters:
    x (numpy.ndarray): 현재 상태
    dt (float): 시간 간격
    
    Returns:
    numpy.ndarray: 다음 상태
    """
    return x  # 선형 상태 전이

def hx(x):
    """
    측정 함수: 두 센서를 통해 상태를 측정한다고 가정.
    
    Parameters:
    x (numpy.ndarray): 현재 상태
    
    Returns:
    numpy.ndarray: 센서1과 센서2의 측정값
    """
    return np.array([x[0], x[0]])  # 두 센서 모두 동일한 상태를 측정

def adjust_R(sensor1_measure, sensor2_measure, base_R1=1.0, base_R2=4.0, factor=0.5, min_R1=0.5, min_R2=2.0):
    """
    센서1과 센서2의 측정값 중 더 큰 값의 센서에 더 큰 비중을 두기 위해 해당 센서의 R을 낮춥니다.
    
    Parameters:
    sensor1_measure (float): 센서1의 측정값
    sensor2_measure (float): 센서2의 측정값
    base_R1 (float): 센서1의 기본 잡음 공분산
    base_R2 (float): 센서2의 기본 잡음 공분산
    factor (float): 줄일 비율 (0 < factor < 1)
    min_R1 (float): 센서1의 최소 잡음 공분산
    min_R2 (float): 센서2의 최소 잡음 공분산
    
    Returns:
    tuple: 조정된 센서1의 R1과 센서2의 R2
    """
    if abs(sensor1_measure) > abs(sensor2_measure):
        # 센서1의 값이 더 크므로 센서1의 R을 낮춤
        adjusted_R1 = max(base_R1 * factor, min_R1)
        adjusted_R2 = base_R2
    elif abs(sensor2_measure) > abs(sensor1_measure):
        # 센서2의 값이 더 크므로 센서2의 R을 낮춤
        adjusted_R1 = base_R1
        adjusted_R2 = max(base_R2 * factor, min_R2)
    else:
        # 값이 같을 경우 기본값 유지
        adjusted_R1 = base_R1
        adjusted_R2 = base_R2
    
    return adjusted_R1, adjusted_R2

def run_ukf(t, true_state, sensor1, sensor2):
    """
    UKF를 사용하여 두 센서 데이터를 합성하여 상태를 추정합니다.
    
    Parameters:
    t (numpy.ndarray): 시간 벡터
    true_state (numpy.ndarray): 진짜 상태
    sensor1 (numpy.ndarray): 센서1의 측정값
    sensor2 (numpy.ndarray): 센서2의 측정값
    
    Returns:
    numpy.ndarray: UKF의 추정값
    """
    # 시그마 포인트 생성
    points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=1.0)

    # 시간 간격 계산
    if len(t) > 1:
        dt = t[1] - t[0]
    else:
        dt = 1.0  # 기본값 설정

    # UKF 초기화
    ukf = UKF(dim_x=1, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)
    ukf.x = np.array([0.0])  # 초기 상태 추정
    ukf.P = np.array([[10.0]])  # 초기 오차 공분산
    ukf.R = np.array([[1.0, 0.0],
                     [0.0, 4.0]])  # 초기 센서1과 센서2의 측정 잡음 공분산
    ukf.Q = np.array([[0.001]])  # 프로세스 잡음 공분산

    # 추정 결과 저장 리스트
    ukf_estimates = []

    for i in range(len(t)):
        # 현재 센서 측정값
        z = np.array([sensor1[i], sensor2[i]])
        
        # 센서1과 센서2의 측정값 중 더 큰 값에 비중을 두기 위해 R을 조정
        adjusted_R1, adjusted_R2 = adjust_R(
            sensor1[i], sensor2[i],
            base_R1=1.0, base_R2=4.0,
            factor=0.5, min_R1=0.5, min_R2=2.0
        )
        ukf.R = np.array([[adjusted_R1, 0.0],
                          [0.0, adjusted_R2]])  # 센서1과 센서2의 측정 잡음 공분산 업데이트
        
        # 예측
        ukf.predict()
        
        # 업데이트
        ukf.update(z)
        
        # 상태 추정 저장
        ukf_estimates.append(ukf.x[0])

    # 추정 결과를 NumPy 배열로 변환
    ukf_estimates = np.array(ukf_estimates)
    
    return ukf_estimates

def plot_results(t, true_state, sensor1, sensor2, ukf_estimates, title, index, save_path, algorithm):
    """
    진짜 상태, 센서1, 센서2, 그리고 UKF 추정값을 시각화합니다.
    
    Parameters:
    t (numpy.ndarray): 시간 벡터
    true_state (numpy.ndarray): 진짜 상태
    sensor1 (numpy.ndarray): 센서1의 측정값
    sensor2 (numpy.ndarray): 센서2의 측정값
    ukf_estimates (numpy.ndarray): UKF의 추정값
    title (str): 그래프 제목
    """

    plt.figure(figsize=(15, 8))
    plt.title(title)
    plt.plot([i for i in range(len(true_state))], true_state, label='Ground Truth', color='green', linewidth=2)
    plt.plot([i for i in range(len(sensor1))][1:], sensor1[1:], label='Frequency model only', color='red', alpha=0.5)
    plt.plot([i for i in range(len(sensor2))][1:], sensor2[1:], label=algorithm, color='blue', alpha=0.5)
    plt.plot([i for i in range(len(ukf_estimates))][1:], ukf_estimates[1:], label='UKF synthesis', color='black', linewidth=2)
    plt.text(int(index[0][1])-2, 70, 0, ha='center', va='bottom', fontsize=15, color='red')
    for value in index:
        plt.axvline(int(int(value[1])), 0, 200, color='red', linestyle='--', linewidth=2)
        plt.text(int(int(value[1]))+2, 70, value[0], ha='center', va='bottom', fontsize=20, color='red')
    plt.legend()
    # plt.grid()
    plt.savefig(save_path)
    plt.show()

def main():
    """
    메인 함수: 데이터 로드, UKF 실행, 결과 시각화를 수행합니다.
    """
    save_base = "D:/home/BCML/drax/PAPER/materials/kalman_integration_improved/"
    
    base_path = "D:/home/BCML/drax/PAPER/materials/fft_version_6sec_good_index/"
    all_entries = os.listdir(base_path)
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(base_path, entry))]
    for folder in folders:
        os.makedirs(f"{save_base}{folder}", exist_ok=True)
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
    
    index = [0,3,5,7,9,'total']
    columns = deepcopy(folders)
    for folder in folders:
        columns.append(f'kalman_integrated_{folder}')
        
    final_table_mae = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_rmse = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_r2 = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count = pd.DataFrame(0.0, index=index, columns=columns)
    final_table_count4r2 = pd.DataFrame(0.0, index=index, columns=columns)
    
    for folder in folders:
        print(f"Processing folder: {folder}")
        data_loader = load_data(base_path, folder)
        
        for t, true_state, sensor1, sensor2, title, index in data_loader:
            print(f"Data loaded: {title}")
            ukf_estimates = run_ukf(t, true_state, sensor1, sensor2)
            index_plot = speeds[title.split('.')[0]]
            save_path = f"{save_base}{folder}/{title.split('.')[0]}_total.png"
            plot_results(t, true_state, sensor1, sensor2, ukf_estimates, title, index, save_path, folder)

            #==========================================================================================

            x_acc_mae = mean_absolute_error(true_state[1:], sensor2[1:])
            x_acc_rmse = np.sqrt(mean_squared_error(true_state[1:], sensor2[1:]))
            x_acc_r2 = r2_score(true_state[1:], sensor2[1:])
            
            with open(f"./materials/kalman_integration_improved/{folder}/performance.txt", 'a') as f:
                f.write(f"{title} x:mae({x_acc_mae})rmse({x_acc_rmse})r2({x_acc_r2})\n")
            
            final_table_mae.loc['total', folder] += x_acc_mae
            final_table_rmse.loc['total', folder] += x_acc_rmse
            if 0 <= x_acc_r2 <= 1:
                final_table_r2.loc['total', folder] += x_acc_r2
                final_table_count4r2.loc['total', folder] += 1
            final_table_count.loc['total', folder] += 1
            
            with open(f"./materials/kalman_integration_improved/{folder}/{title}_time_index.txt", 'a') as f:
                for value in index:
                    f.write(f"{value[0]}_{int(value[1])}\n")
            
            previous_point = 0
            for value in index:
                x_acc_mae = mean_absolute_error(true_state[previous_point:int(value[1])], sensor2[previous_point:int(value[1])])
                x_acc_rmse = np.sqrt(mean_squared_error(true_state[previous_point:int(value[1])], sensor2[previous_point:int(value[1])]))
                x_acc_r2 = r2_score(true_state[previous_point:int(value[1])], sensor2[previous_point:int(value[1])])
                previous_point = int(value[1])
                
                with open(f"./materials/kalman_integration_improved/{folder}/performance.txt", 'a') as f:
                    f.write(f"{title}speed({value[0]}) x:mae({x_acc_mae})rmse({x_acc_rmse})r2({x_acc_r2})\n")
                
                final_table_mae.loc[int(value[0]), folder] += x_acc_mae
                final_table_rmse.loc[int(value[0]), folder] += x_acc_rmse
                if 0 <= x_acc_r2 <= 1:
                    final_table_r2.loc[int(value[0]), folder] += x_acc_r2
                    final_table_count4r2.loc[int(value[0]), folder] += 1
                final_table_count.loc[int(value[0]), folder] += 1
                
            #==========================================================================================
                
            total_results_acc_mae = mean_absolute_error(true_state[1:], ukf_estimates[1:])
            total_results_acc_rmse = np.sqrt(mean_squared_error(true_state[1:], ukf_estimates[1:]))
            total_results_acc_r2 = r2_score(true_state[1:], ukf_estimates[1:])
            
            with open(f"./materials/kalman_integration_improved/{folder}/performance.txt", 'a') as f:
                f.write(f"{title} freq:mae({total_results_acc_mae})rmse({total_results_acc_rmse})r2({total_results_acc_r2})\n")

            final_table_mae.loc['total', f'kalman_integrated_{folder}'] += total_results_acc_mae
            final_table_rmse.loc['total', f'kalman_integrated_{folder}'] += total_results_acc_rmse
            if 0 <= total_results_acc_r2 <= 1:
                final_table_r2.loc['total', f'kalman_integrated_{folder}'] += total_results_acc_r2
                final_table_count4r2.loc['total', f'kalman_integrated_{folder}'] += 1
            final_table_count.loc['total', f'kalman_integrated_{folder}'] += 1

            previous_point = 0
            
            for value in index:
                total_results_acc_mae = mean_absolute_error(true_state[previous_point:int(value[1])], ukf_estimates[previous_point:int(value[1])])
                total_results_acc_rmse = np.sqrt(mean_squared_error(true_state[previous_point:int(value[1])], ukf_estimates[previous_point:int(value[1])]))
                total_results_acc_r2 = r2_score(true_state[previous_point:int(value[1])], ukf_estimates[previous_point:int(value[1])])
                previous_point = int(value[1])
                
                with open(f"./materials/kalman_integration_improved/{folder}/performance.txt", 'a') as f:
                    f.write(f"{title}speed({value[0]}) freq:mae({total_results_acc_mae})rmse({total_results_acc_rmse})r2({total_results_acc_r2})\n")
                
                final_table_mae.loc[int(value[0]), f'kalman_integrated_{folder}'] += x_acc_mae
                final_table_rmse.loc[int(value[0]), f'kalman_integrated_{folder}'] += x_acc_rmse
                if 0 <= total_results_acc_r2 <= 1:
                    final_table_r2.loc[int(value[0]), f'kalman_integrated_{folder}'] += total_results_acc_r2
                    final_table_count4r2.loc[int(value[0]), f'kalman_integrated_{folder}'] += 1
                final_table_count.loc[int(value[0]), f'kalman_integrated_{folder}'] += 1
    
    final_table_mae = final_table_mae/final_table_count
    final_table_rmse = final_table_rmse/final_table_count
    final_table_r2 = final_table_r2/final_table_count4r2
    
    final_table_mae.to_excel(f"./materials/kalman_integration_improved/mae_table.xlsx")
    final_table_rmse.to_excel(f"./materials/kalman_integration_improved/rmse_table.xlsx")
    final_table_r2.to_excel(f"./materials/kalman_integration_improved/r2_table.xlsx")
    
    ax = sns.heatmap(final_table_mae)
    plt.title("total mae")
    plt.savefig(f"./materials/kalman_integration_improved/mae_table.png")
    plt.show()
    
    ax = sns.heatmap(final_table_rmse)
    plt.title("total rmse")
    plt.savefig(f"./materials/kalman_integration_improved/rmse_table.png")
    plt.show()
    
    ax = sns.heatmap(final_table_r2, vmin=0.0, vmax=1.0)
    plt.title("total r2")
    plt.savefig(f"./materials/kalman_integration_improved/r2_table.png")
    plt.show()
if __name__ == "__main__":
    main()