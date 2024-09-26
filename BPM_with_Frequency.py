# 사용법이 조금 조악합니다...
# 수정 중이라 수정되고 나면 드라이브에 업데이트하겠습니다.


from copy import deepcopy
import math
import cv2
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class non_DNN_model():
    def __init__(self, num_window=30, num_step=30, num_frequency_mean_ewma=3, num_frequency_smoothing_ewma=3, num_frequency_diff_ewma=3, num_bpm_ewma=10, 
                alpha_4_frequency_window=0.5, alpha_4_frequency_diff_main=0.5, alpha_4_frequency_diff_end=0.5, alpha_4_frequency_diff_start=0.5, alpha_4_bpm_output=0.3, 
                exp_base_curve=2, exp_power_curve=-0.5, exp_base_scaling=2, exp_power_scaling=-0.5, scaling_factor=0.3, bias=11,
                sigmoid_power_coefficient=0.2, sigmoid_power_constant=-10, sigmoid_numerator=1.1):
    
        self.num_window = num_window
        self.num_step = num_step
        self.num_fme = num_frequency_mean_ewma
        self.num_fse = num_frequency_smoothing_ewma
        self.num_fde = num_frequency_diff_ewma
        self.num_bpm_ewma = num_bpm_ewma
        
        self.a4fw = alpha_4_frequency_window
        self.a4fdm = alpha_4_frequency_diff_main
        self.a4fde = alpha_4_frequency_diff_end
        self.a4fds = alpha_4_frequency_diff_start
        self.a4bo = alpha_4_bpm_output
        
        self.exp_base_curve = exp_base_curve
        self.exp_power_curve = exp_power_curve
        self.exp_base_scaling = exp_base_scaling
        self.exp_power_scaling = exp_power_scaling
        self.bias = bias
        
        self.list_window = []
        self.list_freq_ewma = []
        self.list_diff_ewma = []
        self.list_bpm_for_step = []
        self.bpm_previous = 0
        self.scaling = scaling_factor
        
        self.sig_coeffi = sigmoid_power_coefficient
        self.sig_const = sigmoid_power_constant
        self.sig_numer = sigmoid_numerator
        
        self.exp_power_curve_idx = 0
        self.run_count = 0
        
        if not 0<self.a4fw<1 or not 0<self.a4fdm<1 or not 0<self.a4fde<1 or not 0<self.a4fds<1:
            raise ValueError("value alpha must be in the range : 0 < alpha < 1")
            
        if exp_power_curve>=0 or exp_power_scaling>=0:
            raise ValueError("power of exponential function must be smaller than 0")
    
    def arithmetic_mean(self, input_series):
        return sum(input_series)/len(input_series)
    
    def ewma_forward(self, input_series, alpha=0.5):
        process_series = input_series
        ewma_output = process_series[0]
        if len(process_series)>1:
            for idx in range(len(process_series)-1):
                ewma_output = alpha * ewma_output + (1-alpha) * process_series[idx+1]
        else:
            pass
        return ewma_output

    def ewma_backward(self, input_series, alpha=0.5):
        process_series = input_series
        ewma_output = process_series[-1]
        if len(process_series)>1:
            for idx in range(len(process_series)-1):
                ewma_output = alpha * ewma_output + (1-alpha) * process_series[-idx-1]
        else:
            pass
        return ewma_output
    
    def harmonic_mean(self, input_series):
        denominator = 0
        for value in input_series:
            denominator += (1/value)
        return len(input_series)/denominator
    
    def geometric_mean(self, input_series):
        return output
    
    def harmonic_mean_ewma(self, input_series, alpha=0.5):
        inverse_list = [1/value for value in input_series]
        return len(input_series)/(self.ewma_forward(inverse_list))
    
    def geometric_mean_ewma(self, input_series, alpha=0.5):
        return output
    
    def extract_feature_freq(self, inputs):
        inputs_sorted = list(deepcopy(inputs))
        inputs_sorted.sort(reverse=True)
        inputs_rank = []
        for i in range(len(inputs)):
            inputs_rank.append(inputs_sorted.index(inputs[i]))
        
        feature_freq_list = []
        for i in range(self.num_fme):
            feature_freq_list.append(inputs_rank.index(i))
        
        feature_freq_out = self.arithmetic_mean(feature_freq_list)  # mean of frequency list : futher work: To use statistical method
        
        return feature_freq_out
    
    def sigmoid(self, x):
        return self.sig_numer/(1+math.exp(self.sig_coeffi*(x+self.sig_const)))
    
    def run(self, input_freq, est_start, est_start_initial_bpm=None):
        
        self.run_count += 1
        
        if est_start:
            if est_start_initial_bpm == None:
                raise ValueError("initial BPM is 'None'")
            self.bpm_previous = est_start_initial_bpm
        
        feature_freq = self.extract_feature_freq(input_freq)
        
        if len(self.list_freq_ewma) == self.num_fse:
            del self.list_freq_ewma[0]
        
        self.list_freq_ewma.append(feature_freq)
        self.list_window.append(self.ewma_forward(self.list_freq_ewma, self.a4fw))
        
        if len(self.list_window) == self.num_window:
            
            diff_end = self.ewma_forward(self.list_window, self.a4fde)
            diff_start = self.ewma_backward(self.list_window, self.a4fds)
            diff_acceleration = diff_end - diff_start

            diff_main = self.ewma_forward(self.list_window, self.a4fdm)
            diff = diff_acceleration * diff_main
            
            if len(self.list_diff_ewma) == self.num_fde:
                del self.list_diff_ewma[0]
            
            self.list_diff_ewma.append(diff)
            diff_final = self.ewma_forward(self.list_diff_ewma)
            
            # Utilizing an exponential function to progressively reduce the rate of change in BPM over frequency
            bpm_output = self.bpm_previous + diff_final * self.scaling * self.exp_base_scaling ** (self.exp_power_scaling * abs(diff_final)) + self.bias
            
            # Utilizing an sigmoid function to progressively reduce the rate of change in BPM over frequency
            # bpm_output = self.bpm_previous + diff_final * self.scaling * self.sigmoid(abs(diff_final))
                  
            del self.list_window[0]
            
            self.bpm_previous = bpm_output
            
            return bpm_output

def fft_half(sig_window):
    s_fft = np.fft.fft(sig_window)
    ampl = abs(s_fft) * (2 / len(s_fft))
    f = np.fft.fftfreq(len(s_fft), 1 / sampling_rate)
    ampl_out = np.array([math.log10(value) for value in ampl])  # 각 주파수의 크기에 로그를 취했습니다. 이 부분이 변경되어서 하이퍼파라미터들로는 결과가 이상할 수 있습니다.

    ampl_out = ampl_out[0:int(n_fft // 2)]
    f = f[0:int(n_fft // 2)]
    return ampl_out, f
        
if __name__ == "__main__":
    
    pass_band = (0.5, 10)
    biquad_section = 10
    sampling_rate = 30  # video FPS
    sos = signal.butter(biquad_section, pass_band, 'bandpass', fs=sampling_rate, output='sos')

    #================================================================ @@@ ================================================================
    # 계산의 각 부분에서 저장될 데이터의 길이를 정합니다.
    #
    # num_window = 함수 fft_half()에 의해 반환된 주파수 값에서 크기가 큰 순으로 num_frequency_mean_ewma 개 만큼 뽑고 평균을 취한 값(feature frequency)을 몇개나 저장할지 정합니다.
    #              예를 들어 영상이 30 FPS일 경우 num_window를 30으로 두면 1초간의 feature frequency가 저장됩니다.
    #
    # num_step = 영상에서 몇개의 프레임마다 심박수를 추정할지 정합니다. 샘플링 역할을 하게 됩니다. 이 기능은 아직 구현되지 않았습니다. -X
    #
    # num_frequency_mean_ewma = feature frequency를 뽑기 위해 크기 순으로 나열한 주파수 성분을  몇개나 뽑을지 정합니다.
    #
    # num_frequency_smoothing_ewma = feature frequency를 뽑고 몇개나 exponential weighted moving average(ewma) 할지 정합니다.
    #
    # num_frequency_diff_ewma = 주파수 변화량으로 계산한 심박수 변화량을 몇개나 ewma 할지 정합니다.
    #
    # num_bpm_ewma = 최종 심박수를 몇개나 ewma 할지 정합니다. 이 기능은 아직 구현되지 않았습니다. -X

    num_window = 30
    num_step = 30
    num_frequency_mean_ewma = 3
    num_frequency_smoothing_ewma = 3
    num_frequency_diff_ewma = 3
    num_bpm_ewma = 10

    #================================================================ @@@ ================================================================
    # ewma각 부분에서의 alpha 값을 정합니다.
    #
    # alpha_4_frequency_window = num_window 길이의 주파수의 ewma에서 alpha 값을 정합니다.
    #
    # alpha_4_frequency_diff_main = 심박수의 변화량을 계산하는 등식 "주파수의 변화량 * 주파수" 에서 "주파수" 계산할 때의 alpha 값을 정합니다.
    #
    # alpha_4_frequency_diff_end = 심박수의 변화량을 계산하는 등식 "주파수의 변화량 * 주파수" 에서 "주파수의 변화량" 중 num_window 길이에서 가장 최근 주파수 값을 계산할 때의 alpha 값을 정합니다.
    #
    # alpha_4_frequency_diff_start = 심박수의 변화량을 계산하는 등식 "주파수의 변화량 * 주파수" 에서 "주파수의 변화량" 중 num_window 길이에서 가장 이전 주파수 값을 계산할 때의 alpha 값을 정합니다.
    #
    # alpha_4_bpm_output = 최종 심박수 ewma의 alpha 값을 정합니다. 아직 구현되지 않았습니다. -X

    alpha_4_frequency_window = 0.5
    alpha_4_frequency_diff_main = 0.5
    alpha_4_frequency_diff_end = 0.5
    alpha_4_frequency_diff_start = 0.5
    alpha_4_bpm_output = 0.3

    #================================================================ @@@ ================================================================
    # 아직 쓰이지 않은 변수입니다.
    # feature frequency 값이 일정할 때 "주파수 변화량 * 주파수" 등식에서 "주파수"의 값을 작게 해주기 위해 고안되었으나, "주파수의 변화량"이 작기 때문에 무의미해졌습니다.
    # 하지만 다른 방식으로 쓰일 수 있을 것 같아 남겨 놓았습니다.
    # 추후 쓰이게 되면 주석을 업데이트 하겠습니다.

    exp_base_curve = 2
    exp_power_curve = -0.5

    #================================================================ @@@ ================================================================
    # 주파수의 크기에 따른 등식에서의 "주파수"의 값의 크기를 바꿔주기 위해 고안된 지수함수의 변수들입니다.
    #
    # 현재 이 파일이 적용되었습니다.
    #
    # exp_base_scaling = 지수함수의 밑입니다.
    #
    # exp_power_scaling = 지수함수의 지수입니다.
    #
    # 이 지수함수의 x로는 feature frequency들로 계산된 최종 주파수값(심박수 변화량)이 들어갑니다.
    #
    # bias = 심박수를 계산할 때 더해지는 바이어스입니다.


    exp_base_scaling = 2
    exp_power_scaling = -0.01
    bias = 11

    #================================================================ @@@ ================================================================
    # 주파수로 계산된 최종 심박수 변화량의 크기를 조절합니다.

    scaling_factor = 0.009

    #================================================================ @@@ ================================================================
    # 주파수의 크기에 따른 등식에서의 "주파수"의 값의 크기를 바꿔주기 위해 고안된 시그모이드 함수의 변수들입니다.
    #
    # 현재 이 파일에 적용되지 않았습니다.
    #
    # sigmoid_power_coefficient = e의 지수에 곱해지는 값입니다.
    #
    # sigmoid_power_constant = e의 지수에 더해지는 값입니다.
    #
    # sigmoid_nuerator = 분자에 들어가는 값입니다.

    sigmoid_power_coefficient = 0.1
    sigmoid_power_constant = -10
    sigmoid_nuerator = 1.4

    model = non_DNN_model(num_window=num_window, num_step=num_step, num_frequency_mean_ewma=num_frequency_mean_ewma,
                          num_frequency_smoothing_ewma=num_frequency_smoothing_ewma, num_frequency_diff_ewma=num_frequency_diff_ewma, num_bpm_ewma=num_bpm_ewma,
                          alpha_4_frequency_window=alpha_4_frequency_window, alpha_4_frequency_diff_main=alpha_4_frequency_diff_main,
                          alpha_4_frequency_diff_end=alpha_4_frequency_diff_end, alpha_4_frequency_diff_start=alpha_4_frequency_diff_start, alpha_4_bpm_output=alpha_4_bpm_output, 
                          exp_base_curve=exp_base_curve, exp_power_curve=exp_power_curve, exp_base_scaling=exp_base_scaling, exp_power_scaling=exp_power_scaling, scaling_factor=scaling_factor, bias=bias
                          sigmoid_power_coefficient=sigmoid_power_coefficient, sigmoid_power_constant=sigmoid_power_constant, sigmoid_numerator=sigmoid_numerator)
    
    
    #================================================================ @@@ ================================================================
    # 비디오 경로와 라벨 경로
    
    video_path = ""
    label_path = ""
    
    #================================================================ @@@ ================================================================
    
    cap = cv2.VideoCapture(video_path)
    label = pd.read_csv(label_path, index_col=0)["BPM"].values
    
    fft_window = 300

    initial_bpm = label[fft_window]  # 초기 bpm값을 라벨로 줍니다. (rPPG 기반 알고리즘으로 정확히 추정했다고 가정합니다.)
    
    roi_list = []
    est_bpms = []
    
    index = 0
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("video end")
            break
        
        if len(roi_list) > fft_window:
            del roi_list[0]
        
        #================================================================ @@@ ================================================================
        # 얼굴 감지 함수를 import해 사용합니다.(face_detection_function) <= 보내주셨던 mtcnn을 붙여보려 했는데 제 환경 문제인지 제대로 작동이 되지 않아 고쳐보는 중입니다.
        
        roi_list.append(face_detection_function(frame))  # Currently, a list consist of only one point is supported 지금은 포인트 하나만 지원됩니다.
        
        #================================================================ @@@ ================================================================
        
        if len(roi_list) == fft_window:
            
            roi_list_bpf = signal.sosfilt(sos, roi_list)
            
            ampl_input, _ = fft_half(roi_list_bpf)
            
            if index == 0:
                video_start = True
                initial_bpm = label_arr[0]
                est_bpm = model.run(ampl_input, video_start, initial_bpm)
            else:
                video_start = False
                est_bpm = model.run(ampl_input, video_start)

            if est_bpm is not None:
                est_bpms.append(est_bpm)
        index += 1

    plt.plot(est_bpms)
    plt.title("estimation result")
    plt.xlabel("frame")
    plt.ylabel("BPM")
    plt.show()
    # est_bpms = prediction result list est_bpms가 최종 결과입니다.
