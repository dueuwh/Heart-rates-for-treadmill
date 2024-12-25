# optimize_frequency_model.py

import numpy as np
import pandas as pd
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import time
import os
from frequency_model import freq_model, load_dataset  # frequency_model.py가 같은 디렉토리에 있어야 합니다.
from sklearn.metrics import mean_absolute_error

# Step 1: 하이퍼파라미터를 params.py에서 복사하여 정의
frequency_model_hyperparameters = {
    "num_window": 30,
    "num_step": 30,
    "num_frequency_mean_ewma": 3,
    "num_frequency_smoothing_ewma": 3,
    "num_frequency_diff_ewma": 3,
    "num_bpm_ewma": 10,
    "alpha_4_frequency_window": 0.5,
    "alpha_4_frequency_diff_main": 0.5,
    "alpha_4_frequency_diff_end": 0.5,
    "alpha_4_frequency_diff_start": 0.5,
    "alpha_4_bpm_output": 0.3,
    "exp_base_curve": 2,
    "exp_power_curve": -1.0,
    "exp_base_scaling": 3.5,
    "exp_power_scaling": -0.6,
    "scaling_factor": 0.3,
    "bias": 0.015,
    "sigmoid_p_coefficient": -1.0,
    "sigmoid_e_constant": 0.92,
    "sigmoid_p_constant": -3.0,
    "sigmoid_e_numerator": 2.0,
    "sigmoid_constant": -1.2,
    "arousal_adapt": 0.01,
    "bpm_delay_window": 200,
}

# Step 2: 검색 공간 정의
def get_configspace():
    cs = CS.ConfigurationSpace()

    # 정수형 하이퍼파라미터
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_window", lower=10, upper=100, default_value=frequency_model_hyperparameters["num_window"]))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_step", lower=10, upper=100, default_value=frequency_model_hyperparameters["num_step"]))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_frequency_mean_ewma", lower=1, upper=10, default_value=frequency_model_hyperparameters["num_frequency_mean_ewma"]))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_frequency_smoothing_ewma", lower=1, upper=10, default_value=frequency_model_hyperparameters["num_frequency_smoothing_ewma"]))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_frequency_diff_ewma", lower=1, upper=10, default_value=frequency_model_hyperparameters["num_frequency_diff_ewma"]))
    # cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_bpm_ewma", lower=5, upper=20, default_value=frequency_model_hyperparameters["num_bpm_ewma"]))
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("bpm_delay_window", lower=100, upper=300, default_value=frequency_model_hyperparameters["bpm_delay_window"]))

    # 실수형 하이퍼파라미터 (0 < alpha < 1)
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("alpha_4_frequency_window", lower=0.1, upper=0.9, default_value=frequency_model_hyperparameters["alpha_4_frequency_window"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("alpha_4_frequency_diff_main", lower=0.1, upper=0.9, default_value=frequency_model_hyperparameters["alpha_4_frequency_diff_main"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("alpha_4_frequency_diff_end", lower=0.1, upper=0.9, default_value=frequency_model_hyperparameters["alpha_4_frequency_diff_end"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("alpha_4_frequency_diff_start", lower=0.1, upper=0.9, default_value=frequency_model_hyperparameters["alpha_4_frequency_diff_start"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("alpha_4_bpm_output", lower=0.1, upper=0.9, default_value=frequency_model_hyperparameters["alpha_4_bpm_output"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("scaling_factor", lower=0.1, upper=1.0, default_value=frequency_model_hyperparameters["scaling_factor"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("bias", lower=0.001, upper=0.05, default_value=frequency_model_hyperparameters["bias"]))
    # cs.add_hyperparameter(CSH.UniformFloatHyperparameter("arousal_adapt", lower=0.001, upper=0.1, default_value=frequency_model_hyperparameters["arousal_adapt"]))

    # 실수형 하이퍼파라미터 (exp 관련)
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("exp_base_curve", lower=1.0, upper=5.0, default_value=frequency_model_hyperparameters["exp_base_curve"]))
    # cs.add_hyperparameter(CSH.UniformFloatHyperparameter("exp_power_curve", lower=-5.0, upper=-0.1, default_value=frequency_model_hyperparameters["exp_power_curve"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("exp_base_scaling", lower=1.0, upper=5.0, default_value=frequency_model_hyperparameters["exp_base_scaling"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("exp_power_scaling", lower=-5.0, upper=-0.1, default_value=frequency_model_hyperparameters["exp_power_scaling"]))

    # 실수형 하이퍼파라미터 (sigmoid 관련)
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("sigmoid_p_coefficient", lower=-5.0, upper=-0.1, default_value=frequency_model_hyperparameters["sigmoid_p_coefficient"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("sigmoid_e_constant", lower=0.1, upper=2.0, default_value=frequency_model_hyperparameters["sigmoid_e_constant"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("sigmoid_p_constant", lower=-5.0, upper=-0.1, default_value=frequency_model_hyperparameters["sigmoid_p_constant"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("sigmoid_e_numerator", lower=1.0, upper=5.0, default_value=frequency_model_hyperparameters["sigmoid_e_numerator"]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter("sigmoid_constant", lower=-5.0, upper=-0.1, default_value=frequency_model_hyperparameters["sigmoid_constant"]))

    return cs

# Step 3: Worker 클래스 정의
class FrequencyModelWorker(Worker):
    def __init__(self, *args, **kwargs):
        super(FrequencyModelWorker, self).__init__(*args, **kwargs)
    
    def compute(self, config, budget, **kwargs):
        """
        BOHB의 각 하이퍼파라미터 설정에 대해 호출되는 함수입니다.
        """
        # 수정된 부분: config가 이미 dict이므로 get_dictionary() 호출 제거
        hyperparams = config

        # frequency_model 초기화
        try:
            model = freq_model(hyperparams)
        except Exception as e:
            print(f"Model initialization failed with error: {e}")
            return ({
                'loss': float('inf'),  # 최악의 경우 무한대를 반환
                'info': {}
            })

        # 데이터 로드
        try:
            total_data, algorithms = load_dataset()
        except Exception as e:
            print(f"Data loading failed with error: {e}")
            return ({
                'loss': float('inf'),
                'info': {}
            })

        # 모델 실행 및 평가 지표 계산 (예시로 MAE 사용)
        mae_list = []

        try:
            for key in total_data.keys():
                for algorithm in algorithms:
                    data = total_data[key][algorithm]
                    preds = data['pred']
                    labels = data['label']

                    if len(preds) != len(labels):
                        # 길이가 다르면 보간하거나 무시
                        min_len = min(len(preds), len(labels))
                        preds = preds[:min_len]
                        labels = labels[:min_len]

                    mae = mean_absolute_error(labels, preds)
                    mae_list.append(mae)
            
            # 평균 MAE 계산
            avg_mae = np.mean(mae_list)

        except Exception as e:
            print(f"Model evaluation failed with error: {e}")
            return ({
                'loss': float('inf'),
                'info': {}
            })

        return ({
            'loss': avg_mae,  # 최소화하려는 손실 함수
            'info': {'avg_mae': avg_mae}
        })

# Step 4: BOHB 옵티마이저 초기화 및 실행
if __name__ == "__main__":
    # ConfigSpace 설정
    configspace = get_configspace()

    # 네임서버 초기화
    NS = hpns.NameServer(run_id='frequency_model_optimization', host='127.0.0.1', port=0)
    NS.start()

    # Worker 초기화
    worker = FrequencyModelWorker(nameserver='127.0.0.1', 
                                  nameserver_port=NS.port, 
                                  run_id='frequency_model_optimization')

    worker.run(background=True)

    # BOHB 옵티마이저 초기화
    bohb = BOHB(configspace=configspace,
                run_id='frequency_model_optimization',
                nameserver='127.0.0.1',
                nameserver_port=NS.port,
                min_budget=1,
                max_budget=10)

    # 최적화 실행
    res = bohb.run(n_iterations=50)

    # 최적화 완료 후 네임서버 및 옵티마이저 종료
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # 최적의 하이퍼파라미터 출력
    try:
        print("Best Configurations:")
        for config, cost, info in res.get_incumbent_id():
            print(config)
    
        # 최적의 하이퍼파라미터 상세 정보
        incumbent = res.get_incumbent_id()
        best_config = res.get_id_configuration(incumbent)
        print("Optimized Hyperparameters:")
        print(best_config)
    
        # 최적의 성능 지표 출력
        print(f"Best Loss (MAE): {best_config['avg_mae']}")
        with open("best_parameters.txt", 'a') as f:
            f.write(best_config['avg_mae'])
    except TypeError:
        print("No best model")