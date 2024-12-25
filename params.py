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
        "exp_power_curve":-1.0,
        "exp_base_scaling":3.5,
        "exp_power_scaling":-0.6,
        "scaling_factor":0.3,
        "bias":0.015,
        "sigmoid_p_coefficient":8.0,
        "sigmoid_e_constant":1.0,
        "sigmoid_p_constant":18.0,
        "sigmoid_e_numerator":2.0,
        "sigmoid_constant":-1.2,
        "arousal_adapt":0.01,
        "bpm_delay_window":200,
        "acc_list_length":900,
        "acc_sign_list_length":900,
        "diff_main_list_length":900,
        "diff_coeff_scaling":0.5,
        "diff_coeff_e_coeff":-1000,
        "diff_coeff_e_constant":7,
        
        }
    
    