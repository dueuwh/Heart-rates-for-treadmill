# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:02:06 2024

@author: ys
"""

import numpy as np

class Params():
    frequency_model_hyperparameters={
        "num_window":30,
        "num_step":30,
        "num_frequency_mean_ewma":3,
        "num_frequency_smoothing_ewma":3,
        "num_frequency_diff_ewma":3,
        "num_bpm_ewma":10,
        "alpha_4_frequency_window":0.5,
        "alpha_4_frequency_diff_main":0.5,
        "alpha_4_frequency_diff_end":0.5,
        "alpha_4_frequency_diff_start":0.5,
        "alpha_4_bpm_output":0.3,
        "exp_base_curve":2,
        "exp_power_curve":-0.1,
        "exp_base_scaling":4.2,
        "exp_power_scaling":-0.5,
        "scaling_factor":0.3,
        "bias":0.015,
        "sigmoid_power_coefficient":0.2,
        "sigmoid_power_constant":-10,
        "sigmoid_numerator":1.1,
        "bpm_delay_window":200
        }
    
    