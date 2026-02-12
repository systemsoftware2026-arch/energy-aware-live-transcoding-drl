##########################################################################################
# Imports
##########################################################################################
import itertools
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO, A2C
from torch import nn

from src.training import train_model

import optuna
#from src.training import objective
#from src.training import callback
import pandas as pd

from collections import defaultdict





##########################################################################################
# Grid search
##########################################################################################
"""
GRID = {
    "algorithm_class": [PPO, A2C],
    "gamma": [0.99, 0.9],
    "learning_rate": [0.0003, 0.001],
    "normalize_env": [True, False],
    "activation_fn": [nn.LeakyReLU, nn.ReLU],
    "net_arch": [
        [256, 128],
        [256, 256],
        [512, 256],
        [512, 256, 128],
    ],
}
"""
GRID = {
    "algorithm_class": [PPO],
    "gamma": [0.99],
    "learning_rate": [0.0003],
    "normalize_env": [True],
    "activation_fn": [nn.LeakyReLU],
    "net_arch": [
        [256, 128],
    ],
}

def hyperparam_generator(grid: dict[str, list]):
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    for combination in combinations:
        yield dict(zip(keys, combination))


##########################################################################################
# Training loop
##########################################################################################
def train_models():
    for hyperparams in hyperparam_generator(GRID):
        train_model(
            total_timesteps=100000,
            verbose=0,
            **hyperparams
        )


##########################################################################################
# Main
##########################################################################################
if __name__ == "__main__":
    model=train_models()
    '''
    trial_num = 50
    study = optuna.create_study(direction='maximize')
    #study.optimize(objective, n_trials=100, callbacks=[callback])
    study.optimize(objective, n_trials=trial_num, callbacks=[callback])

    print(f'Best value: {study.best_value}')
    print(f'Best params: {study.best_params}')

    trial_results = []
    for trial in study.trials:
        trial_results.append((trial.value, trial.params))

    clip_range_values = defaultdict(list)

    for value, params in trial_results:
        clip_range_values[params['clip_range']].append(value)

    for clip_range, values in clip_range_values.items():
        avg_value = sum(values) / len(values)
        print(f'Clip range {clip_range}: 평균 value = {avg_value:.4f}')
    '''

    '''
    # trial 결과를 값 기준으로 정렬하고 상위 5개의 결과를 출력
    trial_results.sort(reverse=True, key=lambda x: x[0])
    #top_5_trials = trial_results[:5]
    top_trials = trial_results[:trial_num]
    
    for i, (value, params) in enumerate(top_trials, start=1):
        print(f'Top {i}: Value = {value}, Params = {params}')
    '''
    