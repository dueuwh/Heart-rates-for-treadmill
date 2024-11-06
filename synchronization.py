import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyEMD import EMD, CEEMDAN
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, stft
from sklearn.ensemble import IsolationForest
import copy
import neurokit2 as nk

save_dir = "D:/home/BCML/drax/PAPER/data/rppg_acc_synchro_whiten/"
ppg_dir = "D:/home/BCML/drax/PAPER/data/results/refined/"
acc_dir = "D:/home/BCML/drax/PAPER/data/coordinates/stationary_kalman/"
label_dir = "D:/home/BCML/drax/PAPER/data/treadmill_dataset/labels/"

ppg_list = os.listdir(ppg_dir)
acc_list = os.listdir(acc_dir)
label_list = os.listdir(label_dir)

"""
    save data sampling rate: 150Hz
"""


def minmax(data):
    data_min = min(data)
    data_max = max(data)
    denominator = data_max - data_min
    return np.array([(value - data_min) / denominator for value in data])


def extract_dominant_frequencies_stft(signal, fs, nperseg):
    f, t, Zxx = stft(signal, fs, nperseg=nperseg)

    magnitude = np.abs(Zxx)

    dominant_frequencies = f[np.argmax(magnitude, axis=0)]

    time_original = np.linspace(0, len(signal) / fs, num=len(dominant_frequencies))
    time_full = np.linspace(0, len(signal) / fs, num=len(signal))
    interpolator = interp1d(time_original, dominant_frequencies, kind='linear', fill_value="extrapolate")
    full_length_frequencies = interpolator(time_full)

    return full_length_frequencies


def iso_detection(data, contamination=0.1):
    iso_forest = IsolationForest(contamination=contamination)
    df = pd.DataFrame(data, columns=['Value'])
    df['Outlier'] = iso_forest.fit_predict(df[['Value']])

    df.loc[df['Outlier'] == -1, 'Value'] = np.nan

    df['Value'] = df['Value'].interpolate(method='linear')
    return df['Value'].to_numpy()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def detect_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    return is_outlier


def extract_dominant_frequencies(signal, sampling_rate, threshold=0.1):
    emd = CEEMDAN()
    imfs = emd.emd(signal)
    dominant_frequencies = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1.0 / sampling_rate))
        if len(instantaneous_frequency) > 0:
            median_freq = np.median(instantaneous_frequency)
            if median_freq > threshold:
                dominant_frequencies.append(median_freq)

    return dominant_frequencies


def extract_full_length_instantaneous_frequencies(signal, sampling_rate):
    emd = EMD()
    imfs = emd.emd(signal)

    full_length_frequencies = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1.0 / sampling_rate))

        # interpolation
        time_original = np.arange(len(instantaneous_frequency))
        time_full = np.arange(len(signal))
        interpolator = interp1d(time_original, instantaneous_frequency, kind='linear', fill_value="extrapolate")
        full_length_frequency = interpolator(time_full)

        full_length_frequencies.append(full_length_frequency)

    return full_length_frequencies


def label_preprocessing(data):
    try:
        hr = data["HR"].dropna(axis=0)
        ecg = data["ECG"].dropna(axis=0)
    except:
        hr = data["hr"].dropna(axis=0)
        try:
            ecg = data["ecg"].dropna(axis=0)
        except:
            ecg = data['ECG'].dropna(axis=0)
    return hr, ecg


def normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


total_data = {}
total_data['acc'] = {}
total_data['ecg'] = {}
total_data['emd_1'] = {}
total_data['emd_2'] = {}
total_data['rppg'] = {}
total_data['stft'] = {}

plot_verbose = False
save_file = False

for file_name in ppg_list:

    figure_save_dir = "C:/Users/ys/Desktop/bcml/drax/IEEE_Access/features/"
    emd_save_dir = figure_save_dir + "frequency/"
    others = figure_save_dir + "Others/"
    ecg_save_dir = figure_save_dir + "ecg/"
    hr_save_dir = figure_save_dir + "hr/"

    if file_name.split('.')[0] not in os.listdir(emd_save_dir):
        os.mkdir(emd_save_dir + file_name.split('.')[0])

    emd_save_dir = emd_save_dir + file_name.split('.')[0] + '/'

    temp_ppg = np.load(ppg_dir + file_name)
    temp_acc = np.load(acc_dir + file_name)[3, 1, :]
    temp_label = pd.read_csv(label_dir + file_name.split('.')[0] + '.csv', index_col=0)
    temp_hr, temp_ecg = label_preprocessing(temp_label)
    temp_ecg = temp_ecg.to_numpy()
    temp_hr = temp_hr.to_numpy()

    temp_ppg_len = len(temp_ppg)
    temp_acc_len = len(temp_acc)
    temp_ecg_len = len(temp_ecg)
    temp_hr_len = len(temp_hr)
    temp_ppg_new = []
    temp_acc_new = []
    temp_ecg_new = []
    temp_hr_new = []

    if temp_ppg_len > temp_acc_len:

        x = np.linspace(0, temp_hr_len, temp_hr_len)
        y = temp_hr
        xp = np.linspace(0, temp_hr_len, temp_acc_len)
        temp_hr_new = np.interp(xp, x, y)

        # hr_interval = temp_acc_len//temp_hr_len
        # hr_index = 0
        # for i in range(temp_acc_len):
        #     if i//hr_interval == 0:
        #         temp_hr_new.append(temp_hr[hr_index])
        #         hr_index += 1
        #     else:
        #         temp_hr_new.append(np.nan)
        raise ValueError("temp_ppg_len > temp_acc_len, this condition is not implemented")

    elif temp_ppg_len < temp_acc_len:

        figure_size = (20, 9)

        x_acc = np.linspace(0, temp_acc_len, temp_acc_len)
        y_acc = temp_acc
        x_acc_xp = np.linspace(0, temp_acc_len, temp_ppg_len)
        temp_acc_new = np.interp(x_acc_xp, x_acc, y_acc)

        x_ecg = np.linspace(0, temp_ecg_len, temp_ecg_len)
        y_ecg = temp_ecg
        x_ecg_xp = np.linspace(0, temp_ecg_len, temp_ppg_len)
        temp_ecg_new = np.interp(x_ecg_xp, x_ecg, y_ecg)

        temp_ppg_new = temp_ppg

        x = np.linspace(0, temp_hr_len, temp_hr_len)
        y = temp_hr
        xp = np.linspace(0, temp_hr_len, temp_ppg_len)
        temp_hr_new = np.interp(xp, x, y)

        # temp_hr_new = normalization(temp_hr_new)
        temp_ecg_new = normalization(temp_ecg_new)
        temp_ppg_new = normalization(temp_ppg_new)
        temp_acc_new = normalization(temp_acc_new)

        temp_ecg_new = iso_detection(temp_ecg_new)
        temp_ppg_new = iso_detection(temp_ppg_new)

        temp_emd = EMD()
        imfs = temp_emd(temp_acc_new)
        t = np.linspace(0, len(temp_hr_new) / 30, len(temp_hr_new))
        dominant_frequencies = extract_full_length_instantaneous_frequencies(temp_acc_new, sampling_rate=30)

        time_points = np.arange(len(temp_hr_new)) / 30


        def sampling_rate_5_time(data):
            x_temp = np.linspace(0, len(data), len(data))
            y_temp = data
            x_temp_xp = np.linspace(0, len(data), len(data) * 5)
            return np.interp(x_temp_xp, x_temp, y_temp)


        temp_ecg_new = sampling_rate_5_time(temp_ecg_new)
        temp_ppg_new = sampling_rate_5_time(temp_ppg_new)
        temp_acc_new = sampling_rate_5_time(temp_acc_new)
        temp_hr_new = sampling_rate_5_time(temp_hr_new)

        if plot_verbose:
            plt.figure(figsize=(figure_size))
            plt.title(f"{file_name} features")
            # plt.plot(temp_hr_new, label="hr", zorder=4)
            plt.plot(time_points, temp_ppg_new, label='rppg')
            # plt.plot(temp_ecg_new, label='ecg')
            plt.plot(time_points, temp_acc_new, label="acc")
            plt.legend()
            plt.xticks(np.arange(0, time_points[-1], 30),
                       [f"{int(t // 60)}:{int(t % 60):02d}" for t in np.arange(0, time_points[-1], 30)])
            plt.xlabel("Time (min:sec)")
            plt.ylabel("Normalized Data Value")
            plt.savefig(f"{others}{file_name.split('.')[0]}.png", format="png")
            plt.show()
            plt.close()

            plt.figure(figsize=(figure_size))
            plt.title(f"{file_name} ecg")
            plt.plot(time_points, temp_ecg_new, label='ecg')
            plt.legend()
            plt.xticks(np.arange(0, time_points[-1], 30),
                       [f"{int(t // 60)}:{int(t % 60):02d}" for t in np.arange(0, time_points[-1], 30)])
            plt.xlabel("Time (min:sec)")
            plt.ylabel("Normalized Data Value")
            plt.savefig(f"{ecg_save_dir}{file_name.split('.')[0]}.png", format="png")
            plt.show()
            plt.close()

            plt.figure(figsize=(figure_size))
            plt.title(f"{file_name} hr")
            plt.plot(time_points, temp_hr_new, label='hr')
            plt.legend()
            plt.xticks(np.arange(0, time_points[-1], 30),
                       [f"{int(t // 60)}:{int(t % 60):02d}" for t in np.arange(0, time_points[-1], 30)])
            plt.xlabel("Time (min:sec)")
            plt.ylabel("Normalized Data Value")
            plt.savefig(f"{hr_save_dir}{file_name.split('.')[0]}.png", format="png")
            plt.show()
            plt.close()

        dominant_frequencies_refined = np.zeros(np.array(dominant_frequencies).shape)
        ax = np.linspace(0, len(dominant_frequencies[0]), len(dominant_frequencies[0]))
        reference_0 = np.zeros(ax.shape)

        for i, frequencies in enumerate(dominant_frequencies):
            refined_freq = iso_detection(copy.deepcopy(frequencies))
            # refined_freq = butter_lowpass_filter(refined_freq, cutoff=20.0, fs=30.0)
            dominant_frequencies_refined[i, :] = refined_freq

        stft_frequencies = extract_dominant_frequencies_stft(temp_acc_new, 30, nperseg=30)
        # stft_frequencies = iso_detection(stft_frequencies)
        # stft_frequencies = normalization(stft_frequencies)
        stft_frequencies = normalization(stft_frequencies)
        stft_frequencies = sampling_rate_5_time(stft_frequencies)

        for i, dominant_frequency in enumerate(dominant_frequencies):
            if plot_verbose:
                temp_dominant_frequencies_refined = sampling_rate_5_time(dominant_frequencies_refined[i, :])
                plt.figure(figsize=(figure_size))
                plt.title(f"{file_name} frequency features")
                plt.plot(time_points, stft_frequencies, label=f"stft frequency", zorder=0)
                # plt.plot(dominant_frequency, label=f"IMF {i+1}", zorder=1)
                plt.plot(time_points, temp_dominant_frequencies_refined, label=f"refined IMF {i + 1}", zorder=2)
                plt.plot(time_points, reference_0, label="axis 0")
                plt.legend()
                plt.xticks(np.arange(0, time_points[-1], 30),
                           [f"{int(t // 60)}:{int(t % 60):02d}" for t in np.arange(0, time_points[-1], 30)])
                plt.xlabel("Time (min:sec)")
                plt.ylabel("Normalized Data Value")
                plt.savefig(f"{emd_save_dir}emd_{i}.png", format="png")
                plt.show()
                plt.close()

            if i in [1, 2]:
                if save_file:
                    np.save(f"{save_dir}{file_name.split('.')[0]}_emd_{i}.npy", temp_dominant_frequencies_refined)
                    print(f"{file_name} synchornized emd_{i} file is saved")
        if save_file:
            np.save(f"{save_dir}{file_name.split('.')[0]}_stft.npy", stft_frequencies)
            print(f"{file_name} synchornized stft file is saved")

    else:
        ecg_gap = temp_ecg_len - temp_acc_len
        ecg_interval = ecg_gap // temp_acc_len

        for i in range(temp_acc_len):
            temp_ecg_new.append(temp_ecg[i * ecg_interval])

        temp_ppg_new = temp_ppg
        temp_acc_new = temp_acc

        x = np.linspace(0, temp_hr_len, temp_hr_len)
        y = temp_hr
        xp = np.linspace(0, temp_hr_len, temp_ppg_len)
        temp_hr_new = np.interp(xp, x, y)
        break

    temp_ppg_new = np.array(temp_ppg_new)
    temp_acc_new = np.array(temp_acc_new)

    if temp_ppg_new.shape == temp_acc_new.shape:
        if save_file:
            np.save(f"{save_dir}{file_name.split('.')[0]}_rppg.npy", temp_ppg_new)
            np.save(f"{save_dir}{file_name.split('.')[0]}_acc.npy", temp_acc_new)
            np.save(f"{save_dir}{file_name.split('.')[0]}_ecg.npy", temp_ecg_new)
            np.save(f"{save_dir}{file_name.split('.')[0]}_hr.npy", temp_hr_new)
            print(f"{file_name} synchornized ppg, acc and ecg files are saved")

