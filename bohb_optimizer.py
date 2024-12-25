# bohb_optimizer.py

import math
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from frequency_model import freq_model, load_dataset
from scipy import signal
from params import Params
import time
from copy import deepcopy
import json
import csv

def get_hyperband_iterations(max_resources, eta=3):
    """
    Calculate the maximum number of Hyperband iterations.
    """
    log_eta = lambda x: math.log(x) / math.log(eta)
    s_max = int(math.floor(log_eta(max_resources)))
    B = (s_max + 1) * max_resources
    return s_max, B

class BOHBOptimizer:
    def __init__(self, objective, search_space, max_resources=81, eta=3):
        """
        Initialize the BOHB optimizer.

        Args:
            objective (callable): The objective function to minimize.
            search_space (dict): The hyperparameter search space.
            max_resources (int): The maximum amount of resources allocated to a single configuration.
            eta (int): The downsampling rate.
        """
        self.objective = objective
        self.search_space = search_space
        self.max_resources = max_resources
        self.eta = eta
        self.s_max, self.B = get_hyperband_iterations(max_resources, eta)
        self.all_trials = []  # To store all trial results

    def optimize(self):
        """
        Execute the BOHB optimization process.

        Returns:
            best_params (dict): The best hyperparameters found.
            best_loss (float): The best loss achieved.
        """
        best_loss = float('inf')
        best_params = None

        for s in reversed(range(self.s_max + 1)):
            n = int(math.ceil(self.B / self.max_resources / (s + 1) / self.eta**s))
            r = self.max_resources * self.eta**(-s)
            T = []
            print(f"Bracket s={s}: n={n}, r={r}")

            # Initial sampling
            for i in range(n):
                study = optuna.create_study(direction='minimize')
                trial = study.ask()
                # Get the parameters for this trial
                params = self.suggest_params(study, trial)
                # Evaluate the objective
                loss = self.objective(params, r)
                study.tell(trial, loss)
                T.append((loss, params))
                self.all_trials.append({"loss": loss, "params": params})
                print(f"Trial {i + 1}/{n}: Loss={loss}")

            # Successive Halving
            for i in range(1, s + 1):
                n_i = int(n / self.eta**i)
                r_i = r * self.eta**i
                print(f"  Successive Halving iteration {i}: n_i={n_i}, r_i={r_i}")

                # Select top n_i configurations
                T = sorted(T, key=lambda x: x[0])[:n_i]
                new_T = []
                for j, (loss, params) in enumerate(T):
                    study = optuna.create_study(direction='minimize')
                    trial = study.ask()
                    updated_loss = self.objective(params, r_i)
                    study.tell(trial, updated_loss)
                    new_T.append((updated_loss, params))
                    self.all_trials.append({"loss": updated_loss, "params": params})
                    print(f"    Sub-Trial {j + 1}/{n_i}: Loss={updated_loss}")
                T = new_T

            # Update best parameters
            if T:
                current_best_loss, current_best_params = min(T, key=lambda x: x[0])
                if current_best_loss < best_loss:
                    best_loss = current_best_loss
                    best_params = current_best_params
                    print(f"  New best loss: {best_loss} with params {best_params}")

        return best_params, best_loss

    def suggest_params(self, study, trial):
        """
        Suggest hyperparameters using Optuna's trial.

        Args:
            study (optuna.study.Study): The Optuna study object.
            trial (optuna.trial.FrozenTrial): The trial object.

        Returns:
            params (dict): The suggested hyperparameters.
        """
        params = {}
        for key, spec in self.search_space.items():
            if spec['type'] == 'int':
                params[key] = trial.suggest_int(key, spec['low'], spec['high'])
            elif spec['type'] == 'float':
                params[key] = trial.suggest_float(
                    key, spec['low'], spec['high'], log=spec.get('log', False)
                )
            # Add other types if necessary
        return params

def objective_function(params):
    """
    Objective function to evaluate a set of hyperparameters.

    Args:
        params (dict): Hyperparameters to evaluate.
        resource (int): Resource allocated to this trial (e.g., number of epochs or data points).

    Returns:
        loss (float): The loss metric to minimize (e.g., MAE).
    """
    # Start with default hyperparameters
    hyperparams = deepcopy(Params.frequency_model_hyperparameters)

    # Overwrite with optimized hyperparameters
    for key in params:
        hyperparams[key] = params[key]

    # Initialize the model with the complete set of hyperparameters
    model = freq_model(hyperparams)

    # Load dataset
    data_total, algorithms = load_dataset()

    # Initialize loss accumulator
    total_mae = 0
    count = 0
    
    fft_window = 300
    pass_band = (0.5, 10)
    biquad_section = 10
    sampling_rate = 30
    sos = signal.butter(biquad_section, pass_band, 'bandpass', fs=sampling_rate, output='sos')

    for key in data_total.keys():
        for algorithm in algorithms:
            # Simulate resource allocation by limiting the number of steps
            preds = data_total[key][algorithm]['pred']
            labels = data_total[key][algorithm]['label']
            
            

            # Calculate MAE for this subset
            mae = mean_absolute_error(labels, preds)
            total_mae += mae
            count += 1

    # Average loss across all subsets
    avg_mae = total_mae / count if count > 0 else float('inf')
    print(f"Evaluated params: {hyperparams}, Resource: {resource}, MAE: {avg_mae}")
    return avg_mae

if __name__ == "__main__":
    # Define the hyperparameter search space
    search_space = {
        "num_window": {"type": "int", "low": 10, "high": 50},
        "new_step": {"type": "int", "low": 10, "high": 50},
        "num_frequency_mean_ewma": {"type": "int", "low":1, "high": 5},
        "num_frequency_smoothing_ewma": {"type": "int", "low":1, "high": 5},
        "num_frequency_diff_ewma": {"type": "int", "low": 1, "high": 5},
        "num_bpm_ewma": {"type": "int", "low": 1, "high": 5},
        
        "alpha_4_frequency_window": {"type": "float", "low": 0.1, "high": 0.9},
        "alpha_4_frequency_diff_main": {"type": "float", "low": 0.1, "high": 0.9},
        "alpha_4_frequency_diff_end": {"type": "float", "low": 0.1, "high": 0.9},
        "alpha_4_frequency_diff_start": {"type": "float", "low": 0.1, "high": 0.9},
        "alpha_4_bpm_output": {"type": "float", "low": 0.1, "high": 0.9},
        
        "exp_base_curve": {"type": "int", "low": 1, "high": 3},
        "exp_power_curve": {"type": "float", "low": -2.0, "high": 2.0},
        "exp_base_scaling": {"type": "float", "low": 1.0, "high": 5.0},
        "exp_power_scaling": {"type": "float", "low": -1.0, "high": 1.0},

        "scaling_factor": {"type": "float", "low": 0.1, "high": 1.0},
        "bias": {"type": "float", "low": 0.0, "high": 0.1},
        
        "sigmoid_power_coefficient": {"type": "float", "low": 0.1, "high": 0.9},
        "sigmoid_power_constant": {"type": "float", "low": -10, "high": 2},
        "sigmoid_numerator": {"type": "float", "low": 1.0, "high": 3.0},
        "bpm_delay_window": {"type": "int", "low":10, "high":200},
        # Add more hyperparameters if needed
    }

    # Initialize the BOHB optimizer
    optimizer = BOHBOptimizer(
        objective=objective_function,
        search_space=search_space,
        max_resources=900,  # Adjust based on your resource definition
        eta=3,
    )

    # Start the optimization process
    start_time = time.time()
    best_hyperparams, best_loss = optimizer.optimize()
    end_time = time.time()

    print("\nOptimization completed.")
    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Loss (MAE): {best_loss}")
    print(f"Total Optimization Time: {end_time - start_time:.2f} seconds")

    # Save the best hyperparameters to a JSON file
    results = {
        "best_hyperparameters": best_hyperparams,
        "best_loss_mae": best_loss,
        "optimization_time_seconds": end_time - start_time
    }

    with open("best_hyperparameters.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Best hyperparameters have been saved to 'best_hyperparameters.json'.")

    # Save all trials to a CSV file
    if optimizer.all_trials:
        csv_file_all = "all_trials.csv"
        fieldnames = list(optimizer.all_trials[0].keys())
        with open(csv_file_all, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for trial in optimizer.all_trials:
                writer.writerow(trial)
        print(f"All trials have been saved to '{csv_file_all}'.")