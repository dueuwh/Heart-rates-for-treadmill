# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 04:09:08 2024

@author: ys
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
# from mango import scheduler, Tuner
from sklearn.metrics import r2_score as r2

def adjusted_r2(r2_input, x_test):
    
    """
    params: x_test = model parameters (2 dimension)
    """
    
    x_len = len(x_test)
    num_variables = x_test.shape[1]
    return 1 - (1-r2_input)*(x_len-1)/(x_len-num_variables-1)

class test():
    def __init__(self, series : np.array, order : int):
        self.order = order
        self.series = series
        self.mean = sum(self.series)/len(self.series)
        self.length = len(series)
    
    def acf(args):
        series = args[0]
        num_order = args[1]
        mean = series.mean()
        cov = np.sum((series[:-num_order] - mean)*(series[num_order:] - mean))
        var = np.sum(np.square(series - mean))
        return cov/var
    
    def pacf(args):
        
        return output
    
    def plot():
        acf = list(map(acf, [[self.series, i] for i in range(self.order)]))
        plt.title("pacf and acf plot")
        # plt.plot(self.pacf, label="pacf")
        plt.plot(self.acf, label="acf")
        plt.legend()
        plt.show()

def line_to_list(in_str):
    return [float(x) for x in in_str.split(' ')[:-1]]

def plotting(name, x_list, x_diff_1, y_list, y_diff_1):    
    plt.figure(figsize=(13, 8))
    plt.title(f"{name} coords plot x and x_diff_1")
    plt.plot(x_list, label="x")
    plt.plot(x_diff_1, label="x_diff_1")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(13, 8))
    plt.title(f"{name} coords plot y and y_diff_1")
    plt.plot(y_list, label="y")
    plt.plot(y_diff_1, label="y_diff_1")
    plt.legend()
    plt.show()

def parameter_optimization(x_list, y_list, num_min, num_max, diff):
    model_x = pm.auto_arima(y = np.array(x_list),
                            d = diff,
                            start_p = num_min, max_p = num_max,
                            start_q = num_min, max_q = num_max,
                            trace = True)
    print("\n====================================================")
    print("x summary")
    print(model_x.summary())
    print("====================================================\n")
    model_y = pm.auto_arima(y = np.array(y_list),
                            d = diff,
                            start_p = num_min, max_p = num_max,
                            start_q = num_min, max_q = num_max,
                            trace = True)
    print("\n====================================================")
    print("y summary")
    model_y.summary()
    print("====================================================\n")

def load_dataset(load_dir, eps):
    x_list = []
    y_list = []
    with open(load_dir+"/x.txt", 'r') as f:
        xreject_vidx = []
        count = 0
        while True:
            line = f.readline()
            if line == "":
                break
            for value in line_to_list(line):
                if value < eps:
                    xreject_vidx.append(count)
                else:
                    x_list.append(value)
                count += 1
    
    with open(load_dir+"/y.txt", 'r') as f:
        yreject_vidx = []
        count = 0
        while True:
            line = f.readline()
            if line == "":
                break
            for value in line_to_list(line):
                if value < eps:
                    yreject_vidx.append(count)
                else:
                    y_list.append(value)
                count += 1
    
    return x_list, y_list

# ARIMA model: (19, 3, 0) search space: (0~30, 0~3, 0~30)
# ARIMA model: (7, 0, 0) search space: (0~30, 0, 0~30)
# ARIMA model: (7, 1, 2) search space: (0~30, 1, 0~30)
# ARIMA model: (7, 1, 2) search space: (0~30, 2, 0~30)
# ARIMA model: (19, 3, 0) search space: (0~30, 3, 0~30)

# Model optmization with 0 value rejecting
# console 1/A diff: 0, console 2/A diff: 2, console 3/A diff:3
# ARIMA model: (10, 0, 2) search space(0~30, 0, 0~30) [(7, 0, 3), (8, 0, 0), (15, 0, 2), (18, 0, 5), (2, 0, 0), (6, 0, 0), (6, 0, 0), (15, 0, 0)]
# ARIMA model: (6, 1, 1) search space(0~30, 1, 0~30) [(12, 1, 0), (5, 1, 3), (13, 1, 1), (7, 1, 0), (3, 1, 3), (3, 1, 1), (2, 1, 1), (4, 1, 1)]
# ARIMA model: (13, 2, 0) search space(0~30, 2, 0~30) [(4, 2, 0), (18, 2, 1), (4, 2, 0), (7, 2, 0), (12, 2, 0), (16, 2, 0), (22, 2, 0), (19, 2, 0)]
# ARIMA model: (17, 3, 0) search space(0~30, 3, 0~30) [(11, 3, 0), (14, 3, 0), (3, 3, 1), (1, 3, 0), (30, 3, 1), (19, 3, 0), (30, 3, 0), (30, 3, 0)]

# ARIMA model: diff order: 0 (15, 0, 2) X 10157.521, (18, 0, 5) Y 13054.519
# ARIMA model: diff order: 1 (13, 1, 1) X 10136.434, (7, 1, 0) Y 13461.533
# ARIMA model: diff order: 2 (4, 2, 0) X 11045.856, () Y 


def arima_train(data_name, order, data, forecast_steps, axis):
    
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    
    forecaststeps = 1
    forecast = model_fit.forecast(steps=forecast_steps)
    
    plt.figure(figsize=(13, 8))
    plt.plot(data, label='Original', zorder=1)
    plt.plot(forecast, label='Forecast', color='red', zorder=2)
    plt.title(f'ARIMA Forecast of {data_name} axis: '+axis)
    plt.xlabel('Frame step (1/30 seconds)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    return model_fit

if __name__ == "__main__":
    
    optimization_run = False
    loaded_data_plotting = False
    training_run = True
    
    eps = 0.001
    
    po_min = 0
    po_max = 60
    diff_order = 2
    
    base_dir = "D:/home/BCML/drax/PAPER/data/coordinates/"
    folder_list = os.listdir(base_dir)
    
    total_dataset = {}
    
    total_model = {}
    
    base_dir = "D:/home/BCML/drax/PAPER/data/coordinates/raw/"
    file_list = os.listdir(base_dir)

    file_start = 8

    x_total = []
    y_total = []

    for sel_file in file_list[file_start:]:
        print(f"\n\n{sel_file} start\n\n")
        raw_empty_points = []
        raw_empty_value = []
        
        anomaly_points = []
        
        with open(base_dir+sel_file, 'r') as f:
            lines = f.readlines()
            total_length = int(lines[-1])
            ldmk_len = len(lines[0].split(','))-1
            print("ldmk_len: ", ldmk_len)
            
            coords = np.zeros((ldmk_len, 2, total_length))
            idx = 0
            line_idx = 0
            while idx < total_length:
                temp_array = np.zeros((ldmk_len, 2))
                try:
                    line_split = lines[line_idx].strip().split(',')
                    # print("line_split: ", line_split)
                except IndexError:
                    print("indexError")
                    raw_empty_points.append(idx)
                    raw_empty_value.append(0)
                    coords[:, :, idx] = np.ones((coords.shape[0], coords.shape[1]))
                    line_idx += 1
                    idx += 1
                    continue
                if "None" in lines[line_idx]:
                    # print("None")
                    raw_empty_points.append(idx)
                    raw_empty_value.append(0)
                    coords[:, :, idx] = np.ones((coords.shape[0], coords.shape[1]))
                    line_idx += 1
                    idx += 1
                    print("no facemesh detected")
                    break
                    
                if len(line_split) <= 1:
                    line_idx += 1
                    continue
                for jdx in range(ldmk_len):
                    temp_point = line_split[jdx].split(' ')
                    # print(f"temp_point: {temp_point}, jdx: {jdx}")
                    y = int(temp_point[0].split('.')[0])
                    x = int(temp_point[1].split('.')[0])
                    temp_array[jdx, :] = np.array([y, x])
                    if x > 1000 or y > 1000:
                        anomaly_points.append([lines[line_idx], x, y, idx, jdx])
                # print("temp_array:\n", temp_array)
                coords[:, :, idx] = temp_array
                idx += 1
                line_idx += 1
                if idx % 100 == 0:
                    print(f"reading {sel_file} {round(idx/total_length * 100, 2)} %")
        print("all good")
        
        # coords.shape: (468, 2, frames)
        
        x_total += coords[4, 0, :].tolist()
        y_total += coords[4, 0, :].tolist()
        
        
    model_x = ARIMA(x_total, order=(13, 1, 1))
    model_y = ARIMA(y_total, order=(7, 1, 0))
    model_fit_x = model_x.fit()
    model_fit_y = model_y.fit()
    
    print("model X\n", model_fit_x.summary())
    print("model Y\n", model_fit_y.summary())