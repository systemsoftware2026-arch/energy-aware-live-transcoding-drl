##########################################################################################
# Imports
##########################################################################################
from typing import Type

import numpy as np
import or_gym
import sys
import time
import optuna
import gc
import random

#sys.path.append('/Users/kkm/Desktop/source_code/or-gym-master/or-gym-master/or_gym/envs/classic_or')
#import knapsack

from gym import ObservationWrapper, spaces

from or_gym.envs.classic_or.knapsack import KnapsackEnv
from or_gym.envs.classic_or.knapsack import BinaryKnapsackEnv
from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv
from or_gym.envs.classic_or.vmpacking import VMPackingEnv
from or_gym.envs.classic_or import knapsack


from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch import nn
from stable_baselines3.common.evaluation import evaluate_policy
from or_gym.utils import create_env
import pandas as pd

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete

##########################################################################################
# Configuration and reprodubility
##########################################################################################
# Set seed for reproducibility

# Specify knapsack maximum weight
#"""
MAX_ST = 2048.0
MAX_BW = 800.0
MAX_POWER = 625.0
#"""

VERSION = 7             # num of version
HVP = 0
MVP = 1
LVP = 2

DAY = 24    # (hour)

HOUR = 3600 # (sec)


GB = 1024               # MB is normal size
TB = 1024 * 1024


##########################################################################################
# Custom wrapper to normalize environment observations
##########################################################################################
class NormalizingWrapper(ObservationWrapper):
    """Environment wrapper to divide observations by their maximum value"""

    def __init__(self, env: VMPackingEnv):
        """Change observation space of wrapped environment

        Args:
            env (BoundedKnapsackEnv): environment to wrap (tailed towards or_gym)
        """
        # Perform default wrapper initialization
        super().__init__(env)

        # Change default observation space to concatenate the three vectors from the
        # default or_gym implementation, and allow for floating point values in the
        # range [0, 1].
        # mssong self.observation_space = spaces.Box(0, 1, shape=(603,), dtype=np.float32) <= 200 
        self.observation_space = spaces.Box(0, 1, shape=(176,), dtype=np.float32)

    def observation(self, observation: np.array):
        """Perform postprocessing on observations emitted by wrapped environment

        Args:
            observation (np.array): observation emitted by knapsack environment

        Returns:
            np.array: transformed observation
        """
        # Convert observation to float to allow division and float output type
        
        #  observation = observation.astype(np.float32)

        print("Observation",observation)

        
        # Normalize Bandwidth
        observation['state'][:,1] = observation['state'][:,1] / MAX_BW
        # Normalize Storage
        observation['state'][:,2] = observation['state'][:,2] / MAX_ST
        # Normalize Power
        observation['state'][:,4:-2] = observation['state'][:,4:-2] / MAX_POWER
        
        """
        observation[:,1] = observation[:,1] / MAX_BW
        # Normalize Storage
        observation[:,2] = observation[:,2] / MAX_ST
        # Normalize Power
        observation[:,4:-2] = observation[:,4:-2] / MAX_POWER
        """

        # Concatenate three vectors emitted by default bounded knapsack environment
        # observation = np.reshape(observation, (603,))  mssong 603 
        observation['state'] = np.reshape(observation['state'], (176,))
       #  print("Observation:",observation)
    

        return observation

def mask_fn(env: KnapsackEnv) ->np.ndarray:


    return env.valid_action_mask()


##########################################################################################
# Training function
##########################################################################################
def train_model(
    algorithm_class: Type[OnPolicyAlgorithm] = PPO,
    gamma: float = 0.99,
    learning_rate: float = 0.0003,
    normalize_env: bool = True,
    activation_fn: Type[nn.Module] = nn.ReLU,
    net_arch=[256, 256],
    total_timesteps: int = 100000,
    verbose: int = 1,
) -> OnPolicyAlgorithm:
    """Train model with logging and checkpointing

    Args:
        algorithm_class (Type[OnPolicyAlgorithm], optional): algorithm class to use.
            Defaults to PPO.
        gamma (float, optional): discount factor to use.
            Defaults to 0.99.
        learning_rate (float, optional): learning rate to use.
            Defaults to 0.0003.
        normalize_env (bool, optional): whether to normalize the observation space.
            Defaults to True.
        activation_fn (Type[nn.Module], optional): activation function to use.
            Defaults to nn.ReLU.
        net_arch (list, optional): shared layer sizes for MLPPolicy.
            Defaults to [256, 256].
        total_timesteps (int, optional): total timesteps to train for.
            Defaults to 150000.
        verbose (int, optional): whether to do extensive logging.
            Defaults to 1.

    Returns:
        OnPolicyAlgorithm: trained model
    """
    # Make environment and apply normalization wrapper if specified
    """
    env_config = {'N': 12,
              'max_weight': 100,
              'item_weights': np.array([1, 12, 2, 1, 4, 10, 30, 50, 10, 20, 30, 23]),
              'item_values': np.array([2, 4, 2, 1, 10, 7, 2, 4, 3, 1, 4, 10, 12, 11]),
              'mask': True}
    env: BinaryKnapsackEnv = or_gym.make(
        "Knapsack-v1", env_config=env_config
    )
    """
   

    #env: BinaryKnapsackEnv = or_gym.make(
    #     "Knapsack-v1", max_weight=MAX_WEIGHT, mask=False
    #)

    env_config={}

    env: KnapsackEnv = or_gym.make("Knapsack-v0") # , env_config=env_config)
    env = ActionMasker(env, mask_fn)



    #if normalize_env:
        #env = NormalizingWrapper(env)

    # Initialize environment by resetting
    env.reset()

    # All run logs will be saved in the logs folder, under a subfolder indicating the
    # hyperparameters of the run
    """
    
    log_path = "logs/"
    run_name = f"{algorithm_class.__name__}-{gamma}-{learning_rate}-{normalize_env}-{activation_fn.__name__}-{net_arch}"

    # Logging
    print(f"\nStarting run {run_name}...")

    # Callback to perform and save intermittent evaluation steps and checkpoint the best
    # model based on the average reward
    eval_callback = EvalCallback(
        eval_env=env,
        n_eval_episodes=100,
        log_path=log_path + run_name + "_1",
        best_model_save_path=log_path + run_name + "_1",
        deterministic=False,
    )

    # Model definition
    model = algorithm_class(
        policy="MlpPolicy",
        env=env,
        gamma=gamma,
        learning_rate=learning_rate,
        policy_kwargs=dict(
            activation_fn=activation_fn,
            net_arch=net_arch,
        ),
        tensorboard_log=log_path,
        verbose=verbose,
    )

    # Model training
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        tb_log_name=run_name,
    )

    """
    """
    model = PPO("MultiInputPolicy",
                env=env,
                verbose=1,
                gamma=0.99,
                policy_kwargs=dict(
                    activation_fn=nn.ReLU,
                    net_arch=(256,256),
                ),
                learning_rate=0.001,tensorboard_log="./test")
    """
    #entropy_coef = 0.001 # base 0.01
    entropy_coef = 0.0 # base 0.01

    #model = DQN("MultiInputPolicy", env, tensorboard_log="./dqn_test", verbose=1)  #MlpPolicy
    #model = MaskablePPO("MultiInputPolicy", env, tensorboard_log="./dqn_test", verbose=1)
    #model = PPO("MultiInputPolicy", env, tensorboard_log="./ppo_test",verbose=1)
    
    # Train the agent and display a progress bar

    #start_time = time.time()

    model = MaskablePPO("MultiInputPolicy", env, tensorboard_log="./dqn_test", ent_coef=entropy_coef, verbose=1)
    model.learn(total_timesteps=int(10000*100), progress_bar=True)          # normally 5000 episodes(?)
    # Save the agent
    #end_time = time.time()

    #print("execution time is : ", end_time - start_time)
    #print("time per episode is : ", (end_time - start_time) / env.get_episode_cnt())
    #exit()
    model.save("test2/test")


    #"""

    # env.sample_action()

    # Stop environment
    # env.close()
    
    model = MaskablePPO.load("test2/test")
    #model = MaskablePPO.load("TC_energy/T45_22")
    #model = MaskablePPO.load("TC_zipf/zipf10/T30_22")
    #model = MaskablePPO.load("TC_version/RVP/T30_22")
    #model = MaskablePPO.load("TC_shuffle/T40_22")


    #model = DQN.load("pop_test/test")
    # model = PPO.load("results_packing")
    
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


    
    #'''
    #here!!!
    N_section = 24
    N_ratio = 10
    PPO_PWVQ = 0
    HPF_PWVQ = 0
    HPTF_PWVQ = 0
    HMPV_PWVQ = 0
    HMPVT_PWVQ = 0
    HPTF_RR_PWVQ = 0
    HMPVT_RR_PWVQ = 0

    PPO_PWVQ_arr = np.zeros((N_section, N_ratio))
    PPO_energy_arr = np.zeros((N_section, N_ratio))

    HPF_PWVQ_arr = np.zeros((N_section, N_ratio))
    HPF_energy_arr = np.zeros((N_section, N_ratio))

    HPTF_PWVQ_arr = np.zeros((N_section, N_ratio))
    HPTF_energy_arr = np.zeros((N_section, N_ratio))

    HMPV_PWVQ_arr = np.zeros((N_section, N_ratio))
    HMPV_energy_arr = np.zeros((N_section, N_ratio))

    HMPVT_PWVQ_arr = np.zeros((N_section, N_ratio))
    HMPVT_energy_arr = np.zeros((N_section, N_ratio))

    HPTF_RR_PWVQ_arr = np.zeros((N_section, N_ratio))
    HPTF_RR_energy_arr = np.zeros((N_section, N_ratio))

    HMPVT_RR_PWVQ_arr = np.zeros((N_section, N_ratio))
    HMPVT_RR_energy_arr = np.zeros((N_section, N_ratio))

    total_energy = 0

    lowest_PWVQ = np.zeros(N_section)
    lowest_energy = np.zeros(N_section)


    for l in range(N_section):
        video_of_section = random.randint(200, 500)     #  video's # is 200~500
        
        env = KnapsackEnv(video_num=video_of_section)
        #env_config = {"video_num": video_of_section}
        #env: KnapsackEnv = or_gym.make("Knapsack-v0", env_config=env_config) # , env_config=env_config)
        #env: KnapsackEnv = or_gym.make("Knapsack-v0") # , env_config=env_config)
        
        env = ActionMasker(env, mask_fn)
        obs, action_masks = env.reset()
        total_energy += env.get_total_energy()

        lowest_PWVQ[l] = env.get_lowest_PWVQ()
        lowest_energy[l] = env.get_lowest_energy()

        #N_values = [200, 250, 300, 350, 400, 450, 500]
            # video_of_section에 가장 가까운 N 값을 찾기
        #closest_N = min(N_values, key=lambda x: abs(x - video_of_section))
        
        
        for nst in range(N_ratio - 1):      # not 100%
            #obs, action_masks = env.reset()
            env.decision_reset()
            #model_path = f"IGSC_journal/E{(nst + 1) * 10}"
            #model_path = f"1015_N{closest_N}/T{(nst + 1) * 10}_22"
            model_path = f"TC_energy/T{(nst + 1) * 10}_22"
            #model_path = f"TC_zipf/zipf08/T{(nst + 1) * 10}_22"
            #model_path = f"TC_version/HVP/T{(nst + 1) * 10}_22"



            #model_path = f"TC_energy/T{(nst + 1) * 25}_22"  # n = 4
            #model_path = f"TC_energy/T{(nst + 1) * 20}_22"  # n = 5
            #model_path = f"TC_energy/T{(nst + 1) * 5}_22"  # n = 20
            
            
            
            model = MaskablePPO.load(model_path)
            
            env.set_E_limit(nst, N_ratio)
            E_limit = env.get_E_limit()
            #print(env.T_limit)
            print("video num is : ",env.get_video_num())
            #print("closest video is : ", closest_N)



            while True:
                action, _states = model.predict(obs, action_masks=mask_fn(env))
                obs, rewards, dones, truncated, action_masks = env.step(action)

                if dones:
                    env.comparison_process()
                    PPO_PWVQ_arr[l, nst] = env.get_PPO_PWQ()
                    PPO_energy_arr[l, nst] = env.get_cumulative_energy()

                    HPF_PWVQ_arr[l, nst] = env.get_HPF_PWQ()
                    HPF_energy_arr[l, nst] = env.get_HPF_cumulative_energy()

                    HPTF_PWVQ_arr[l, nst] = env.get_HPTF_PWQ()
                    HPTF_energy_arr[l, nst] = env.get_HPTF_cumulative_energy()

                    HMPV_PWVQ_arr[l, nst] = env.get_HMPV_PWQ()
                    HMPV_energy_arr[l, nst] = env.get_HMPV_cumulative_energy()

                    HMPVT_PWVQ_arr[l, nst] = env.get_HMPVT_PWQ()
                    HMPVT_energy_arr[l, nst] = env.get_HMPVT_cumulative_energy()

                    HPTF_RR_PWVQ_arr[l, nst] = env.get_HPTF_RR_PWQ()
                    HPTF_RR_energy_arr[l, nst] = env.get_HPTF_RR_cumulative_energy()
                    HMPVT_RR_PWVQ_arr[l, nst] = env.get_HMPVT_RR_PWQ()
                    HMPVT_RR_energy_arr[l, nst] = env.get_HMPVT_cumulative_energy()

                    break

            del model
            gc.collect()

        PPO_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()
        HPF_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()
        HPTF_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()
        HMPV_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()
        HMPVT_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()
        HPTF_RR_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()
        HMPVT_RR_PWVQ_arr[l, N_ratio - 1] = env.get_optimal_PWQ()

        PPO_energy_arr[l, N_ratio - 1] = env.get_total_energy()
        HPF_energy_arr[l, N_ratio - 1] = env.get_total_energy()
        HPTF_energy_arr[l, N_ratio - 1] = env.get_total_energy()
        HMPV_energy_arr[l, N_ratio - 1] = env.get_total_energy()
        HMPVT_energy_arr[l, N_ratio - 1] = env.get_total_energy()
        HPTF_RR_energy_arr[l, N_ratio - 1] = env.get_total_energy()
        HMPVT_RR_energy_arr[l, N_ratio - 1] = env.get_total_energy()
    

    


    requests = np.array([135, 164, 178, 213, 275, 272, 260, 224, 261, 253, 219, 227, 
                     178, 184, 258, 441, 541, 536, 341, 228, 224, 187, 179, 179])

    # 요청 수를 0.1~0.8 사이로 정규화
    mean_requests = np.mean(requests)
    std_deviation = 50
    num_samples = len(requests) 
    generated_requests = np.random.normal(mean_requests, std_deviation, num_samples)


    min_value = 0.1
    max_value = 1.0
    normalized_generated_requests = (generated_requests - np.min(generated_requests)) / (np.max(generated_requests) - np.min(generated_requests))
    #normalized_workload = normalized_generated_requests * (max_value - min_value) + min_value

    
    
    


    
    
    normalized_workload = [0.1       , 0.41533338, 0.3028857, 0.88180195, 0.725     ,
                            0.87353143, 0.55      , 0.26646857, 0.175     , 0.46819805,
                            0.33971143, 0.88466662, 1.        , 0.98466662, 0.83971143,
                            0.86819805, 0.775     , 0.66646857, 0.15      , 0.23353143,
                            0.225     , 0.63180195, 0.86028857, 0.71533338]
    
    '''
    normalized_workload = [0.1       , 0.21533338, 0.3028857, 0.38180195, 0.325     ,
                            0.37353143, 0.35      , 0.16646857, 0.275     , 0.26819805,
                            0.33971143, 0.18466662, 1.        , 0.28466662, 0.23971143,
                            0.36819805, 0.375     , 0.36646857, 0.35      , 0.13353143,
                            0.225     , 0.13180195, 0.26028857, 0.21533338]
    '''
    normalized_workload = np.array(normalized_workload)
    #print(normalized_workload)
    #exit()

    PPO_PWVQ_arr = PPO_PWVQ_arr * normalized_workload[:, np.newaxis]
    HPF_PWVQ_arr = HPF_PWVQ_arr * normalized_workload[:, np.newaxis]
    HPTF_PWVQ_arr = HPTF_PWVQ_arr * normalized_workload[:, np.newaxis]
    HMPV_PWVQ_arr = HMPV_PWVQ_arr * normalized_workload[:, np.newaxis]
    HMPVT_PWVQ_arr = HMPVT_PWVQ_arr * normalized_workload[:, np.newaxis]
    HPTF_RR_PWVQ_arr = HPTF_RR_PWVQ_arr * normalized_workload[:, np.newaxis]
    HMPVT_RR_PWVQ_arr = HMPVT_RR_PWVQ_arr * normalized_workload[:, np.newaxis]

    
  
        

    
    
    ENERGY_LIMIT = total_energy * 0.3  # total energy percent limit
    OPTIMAL_PWVQ = np.sum(PPO_PWVQ_arr[:, N_ratio - 1])

    datasets = [
        {'PWVQ_arr': PPO_PWVQ_arr, 'energy_arr': PPO_energy_arr, 'PWVQ_sum': np.sum(PPO_PWVQ_arr[:, 0]), 'energy_sum': np.sum(PPO_energy_arr[:, 0])},
        {'PWVQ_arr': HPF_PWVQ_arr, 'energy_arr': HPF_energy_arr, 'PWVQ_sum': np.sum(HPF_PWVQ_arr[:, 0]), 'energy_sum': np.sum(HPF_energy_arr[:, 0])},
        {'PWVQ_arr': HPTF_PWVQ_arr, 'energy_arr': HPTF_energy_arr, 'PWVQ_sum': np.sum(HPTF_PWVQ_arr[:, 0]), 'energy_sum': np.sum(HPTF_energy_arr[:, 0])},
        {'PWVQ_arr': HMPV_PWVQ_arr, 'energy_arr': HMPV_energy_arr, 'PWVQ_sum': np.sum(HMPV_PWVQ_arr[:, 0]), 'energy_sum': np.sum(HMPV_energy_arr[:, 0])},
        {'PWVQ_arr': HMPVT_PWVQ_arr, 'energy_arr': HMPVT_energy_arr, 'PWVQ_sum': np.sum(HMPVT_PWVQ_arr[:, 0]), 'energy_sum': np.sum(HMPVT_energy_arr[:, 0])},
        {'PWVQ_arr': HPTF_RR_PWVQ_arr, 'energy_arr': HPTF_RR_energy_arr, 'PWVQ_sum': np.sum(HPTF_RR_PWVQ_arr[:, 0]), 'energy_sum': np.sum(HPTF_RR_energy_arr[:, 0])},
        {'PWVQ_arr': HMPVT_RR_PWVQ_arr, 'energy_arr': HMPVT_RR_energy_arr, 'PWVQ_sum': np.sum(HMPVT_RR_PWVQ_arr[:, 0]), 'energy_sum': np.sum(HMPVT_RR_energy_arr[:, 0])}
    ]

    PWVQ_results = []
    



    for data in datasets:               # MCKP
        # 데이터셋에서 변수 초기화

        PWVQ = data['PWVQ_sum']
        energy = data['energy_sum']
        current_energy = energy

        numerator = data['PWVQ_arr'][:, 0:] - data['PWVQ_arr'][:, [0]]
        denominator = data['energy_arr'][:, 0:] - data['energy_arr'][:, [0]] 
        

        MCKP_2d_arr = np.zeros_like(denominator, dtype=float)
        np.divide(numerator, denominator, out=MCKP_2d_arr, where=denominator != 0)
        #print(MCKP_2d_arr)
        
        
        now_section_nst = np.zeros(N_section).astype(int)

        #indices = np.array(np.unravel_index(np.argsort(-MCKP_2d_arr.ravel()), MCKP_2d_arr.shape)).T

        while current_energy <= ENERGY_LIMIT:
            if np.all(MCKP_2d_arr == 0):
                break
            max_index = np.argmax(MCKP_2d_arr)
            max_index_2d = np.unravel_index(max_index, MCKP_2d_arr.shape)
            section = max_index_2d[0]
            nst = max_index_2d[1]
            MCKP_2d_arr[section, nst] = 0

            if nst < now_section_nst[section]:
                continue

            now_energy = (data['energy_arr'][section, nst] - data['energy_arr'][section, now_section_nst[section]])
            if current_energy + now_energy > ENERGY_LIMIT:
                continue
            
            current_energy += now_energy
            PWVQ += (data['PWVQ_arr'][section, nst] - data['PWVQ_arr'][section, now_section_nst[section]])
            now_section_nst[section] = nst

            if nst != N_ratio - 1:
                for n in range(nst + 1, N_ratio):
                    MCKP_2d_arr[section, n] = (data['PWVQ_arr'][section, n] - data['PWVQ_arr'][section, nst]) / (data['energy_arr'][section, n] - data['energy_arr'][section, nst])
                    if MCKP_2d_arr[section, n] < 0:
                        MCKP_2d_arr[section, n] = 0

                
        PWVQ_results.append(PWVQ)
    


    # PPO, HPTF_RR, HMPVT_RR
    
    PPO_PWVQ = PWVQ_results[0]
    HPF_PWVQ = PWVQ_results[1]
    HPTF_PWVQ = PWVQ_results[2]
    HMPV_PWVQ = PWVQ_results[3]
    HMPVT_PWVQ = PWVQ_results[4]
    HPTF_RR_PWVQ = PWVQ_results[5]
    HMPVT_RR_PWVQ = PWVQ_results[6]

    print("MCKP results!!!")



    print("PPO : ", PPO_PWVQ)
    print("HPF : ", HPF_PWVQ)
    print("HPTF : ", HPTF_PWVQ)
    print("HMPV : ", HMPV_PWVQ)
    print("HMPVT : ", HMPVT_PWVQ)
    print("HPTF_RR : ", HPTF_RR_PWVQ)
    print("HMPVT_RR : ", HMPVT_RR_PWVQ)

    max_PWVQ = max(HPTF_RR_PWVQ, HMPVT_RR_PWVQ)
    max_PWVQ = max(max_PWVQ, HPTF_PWVQ)
    print("(%) : ", ((PPO_PWVQ - max_PWVQ) / PPO_PWVQ) * 100)
    print("against optimal value(%) : ", ((OPTIMAL_PWVQ - PPO_PWVQ) / OPTIMAL_PWVQ) * 100)
    print("optimal PWVQ is : ", OPTIMAL_PWVQ)

    #exit()

    # greedy algoroithm (PWVQ*lambda / energy)

    greedy_results = []

    for data in datasets:
        PWVQ = data['PWVQ_sum']
        energy = data['energy_sum']
        current_energy = energy

        numerator = data['PWVQ_arr'][:, 1:]
        denominator = data['energy_arr'][:, 1:]
        greedy_arr = np.zeros_like(denominator, dtype=float)
        np.divide(numerator, denominator, out=greedy_arr, where=denominator != 0)

        now_section_nst = np.zeros(N_section).astype(int)
        indices = np.array(np.unravel_index(np.argsort(-greedy_arr.ravel()), greedy_arr.shape)).T

        for section, nst in indices:
            nst = nst + 1
            weight = (data['energy_arr'][section, nst] - data['energy_arr'][section, now_section_nst[section]])
            value = (data['PWVQ_arr'][section, nst] - data['PWVQ_arr'][section, now_section_nst[section]])
            if energy + weight <= ENERGY_LIMIT:
                energy += weight
                PWVQ += value
                now_section_nst[section] = nst

        greedy_results.append(PWVQ)

    PPO_PWVQ = greedy_results[0]
    HPF_PWVQ = greedy_results[1]
    HPTF_PWVQ = greedy_results[2]
    HMPV_PWVQ = greedy_results[3]
    HMPVT_PWVQ = greedy_results[4]
    HPTF_RR_PWVQ = greedy_results[5]
    HMPVT_RR_PWVQ = greedy_results[6]

    print("greedy results!!!")

    print("PPO : ", PPO_PWVQ)
    print("HPF : ", HPF_PWVQ)
    print("HPTF : ", HPTF_PWVQ)
    print("HMPV : ", HMPV_PWVQ)
    print("HMPVT : ", HMPVT_PWVQ)
    print("HPTF_RR : ", HPTF_RR_PWVQ)
    print("HMPVT_RR : ", HMPVT_RR_PWVQ)

    max_PWVQ = max(HPTF_RR_PWVQ, HMPVT_RR_PWVQ)
    max_PWVQ = max(max_PWVQ, HPTF_PWVQ)
    print("(%) : ", ((PPO_PWVQ - max_PWVQ) / PPO_PWVQ) * 100)
    print("against optimal value(%) : ", ((OPTIMAL_PWVQ - PPO_PWVQ) / OPTIMAL_PWVQ) * 100)
    print("optimal PWVQ is : ", OPTIMAL_PWVQ)



    # greedy algoroithm (lambda / energy)

    lambda_results = []

    for data in datasets:
        PWVQ = data['PWVQ_sum']
        energy = data['energy_sum']
        current_energy = energy

        new_arr = np.tile(normalized_workload, (N_ratio, 1)).T

        numerator = new_arr[:, 1:]
        #numerator = data['PWVQ_arr'][:, 1:]
        denominator = np.tile(np.arange(2, N_ratio + 1), (numerator.shape[0], 1))
        #denominator = data['energy_arr'][:, 1:]
        lambda_arr = np.zeros_like(denominator, dtype=float)
        np.divide(numerator, denominator, out=lambda_arr, where=denominator != 0)

        now_section_nst = np.zeros(N_section).astype(int)
        indices = np.array(np.unravel_index(np.argsort(-lambda_arr.ravel()), greedy_arr.shape)).T

        for section, nst in indices:
            nst = nst + 1
            weight = (data['energy_arr'][section, nst] - data['energy_arr'][section, now_section_nst[section]])
            value = (data['PWVQ_arr'][section, nst] - data['PWVQ_arr'][section, now_section_nst[section]])
            if energy + weight <= ENERGY_LIMIT:
                energy += weight
                PWVQ += value
                now_section_nst[section] = nst

        lambda_results.append(PWVQ)

    PPO_PWVQ = lambda_results[0]
    HPF_PWVQ = lambda_results[1]
    HPTF_PWVQ = lambda_results[2]
    HMPV_PWVQ = lambda_results[3]
    HMPVT_PWVQ = lambda_results[4]
    HPTF_RR_PWVQ = lambda_results[5]
    HMPVT_RR_PWVQ = lambda_results[6]

    print("lambda results!!!")

    print("PPO : ", PPO_PWVQ)
    print("HPF : ", HPF_PWVQ)
    print("HPTF : ", HPTF_PWVQ)
    print("HMPV : ", HMPV_PWVQ)
    print("HMPVT : ", HMPVT_PWVQ)
    print("HPTF_RR : ", HPTF_RR_PWVQ)
    print("HMPVT_RR : ", HMPVT_RR_PWVQ)

    max_PWVQ = max(HPTF_RR_PWVQ, HMPVT_RR_PWVQ)
    max_PWVQ = max(max_PWVQ, HPTF_PWVQ)
    print("(%) : ", ((PPO_PWVQ - max_PWVQ) / PPO_PWVQ) * 100)
    print("against optimal value(%) : ", ((OPTIMAL_PWVQ - PPO_PWVQ) / OPTIMAL_PWVQ) * 100)
    print("optimal PWVQ is : ", OPTIMAL_PWVQ)



    # uniform algorithm

    uniform_results = []
    

    for data in datasets:           # uniform (균등 분배)
        PWVQ = data['PWVQ_sum']
        energy = data['energy_sum']
        current_energy = energy


        #PWVQ = 0
        #energy = 0
        #current_energy = energy

        cur_section_idx = 0
        cur_energy_budget_idx = 1
        while current_energy <= ENERGY_LIMIT:
            current_energy += (data['energy_arr'][cur_section_idx, cur_energy_budget_idx] - data['energy_arr'][cur_section_idx, cur_energy_budget_idx - 1])
            if current_energy > ENERGY_LIMIT:
                break
            PWVQ += (data['PWVQ_arr'][cur_section_idx, cur_energy_budget_idx] - data['PWVQ_arr'][cur_section_idx, cur_energy_budget_idx - 1])
            if cur_section_idx == N_section - 1:
                cur_section_idx = 0
                if cur_energy_budget_idx < N_ratio - 1:
                    cur_energy_budget_idx += 1
                continue
            cur_section_idx += 1
        
        uniform_results.append(PWVQ)
    
    PPO_PWVQ = uniform_results[0]
    HPF_PWVQ = uniform_results[1]
    HPTF_PWVQ = uniform_results[2]
    HMPV_PWVQ = uniform_results[3]
    HMPVT_PWVQ = uniform_results[4]
    HPTF_RR_PWVQ = uniform_results[5]
    HMPVT_RR_PWVQ = uniform_results[6]

    print("UNIFORM results!!!")


    print("PPO : ", PPO_PWVQ)
    print("HPF : ", HPF_PWVQ)
    print("HPTF : ", HPTF_PWVQ)
    print("HMPV : ", HMPV_PWVQ)
    print("HMPVT : ", HMPVT_PWVQ)
    print("HPTF_RR : ", HPTF_RR_PWVQ)
    print("HMPVT_RR : ", HMPVT_RR_PWVQ)

    max_PWVQ = max(HPTF_RR_PWVQ, HMPVT_RR_PWVQ)
    max_PWVQ = max(max_PWVQ, HPTF_PWVQ)
    print("(%) : ", ((PPO_PWVQ - max_PWVQ) / PPO_PWVQ) * 100)
    print("against optimal value(%) : ", ((OPTIMAL_PWVQ - PPO_PWVQ) / OPTIMAL_PWVQ) * 100)
    print("optimal PWVQ is : ", OPTIMAL_PWVQ)

    
    # PWVQ / energy 정렬 후 check 배열 통해서 각 section마다 하나만 선택하게끔.

    
            
            
    exit()


    #end!!!
    #'''
    

    episode_cnt = env.get_episode_cnt()
    cumulative_time = env.get_cumulative_time()
    episode_time = cumulative_time / episode_cnt


    results_list=[]
    _collected_items = []

    check = 0

    PPO_PWQ = 0
    HPF_PWQ = 0
    HPTF_PWQ = 0
    HMPV_PWQ = 0
    HMPVT_PWQ = 0
    HPTF_RR_PWQ = 0
    HMPVT_RR_PWQ = 0


    optimal_PWQ = 0

    P_trans = 174       # transcoding Power (Wh)
    
    N_video = 500
    cnt = 0
    #version_list = np.zeros((N_video ,VERSION))


    for i in range(30):
        video_of_section = random.randint(200, 500)     #  video's # is 200~500
        #video_of_section = 1000
        env = KnapsackEnv(video_num=video_of_section)
        env = ActionMasker(env, mask_fn)
        env.decision_reset()



        obs, action_masks = env.reset()
        while True:
            action, _states = model.predict(obs, action_masks=mask_fn(env))
            obs, rewards, dones, truncated, action_masks = env.step(action)

            if dones:
                env.comparison_process()
                PPO_PWQ += env.get_PPO_PWQ()
                HPF_PWQ += env.get_HPF_PWQ()
                HPTF_PWQ += env.get_HPTF_PWQ()
                HMPV_PWQ += env.get_HMPV_PWQ()
                HMPVT_PWQ += env.get_HMPVT_PWQ()
                HPTF_RR_PWQ += env.get_HPTF_RR_PWQ()
                HMPVT_RR_PWQ += env.get_HMPVT_RR_PWQ()


                optimal_PWQ += env.get_optimal_PWQ()
                print("now video number is : ", env.get_video_num())

                #version_list = env.get_ver_list()


                
                break
        #print("total PWQ is : ", env.get_current_PWQ())
        #print("total PWQ threshold is : ", env.get_PWO_threshold())
        #print("transcoding time is : ",env.get_transcoding_time())
    
    

    
    PPO_PWQ = PPO_PWQ / 30
    HPF_PWQ = HPF_PWQ / 30
    HPTF_PWQ = HPTF_PWQ / 30
    HMPV_PWQ = HMPV_PWQ / 30
    HMPVT_PWQ = HMPVT_PWQ / 30

    HPTF_RR_PWQ = HPTF_RR_PWQ / 30
    HMPVT_RR_PWQ = HMPVT_RR_PWQ / 30
    optimal_PWQ = optimal_PWQ / 30

    print("PPO PWQ is : ", PPO_PWQ)
    print("HPF PWQ is : ", HPF_PWQ)
    print("HPTF PWQ is : ", HPTF_PWQ)
    print("HMPV PWQ is : ", HMPV_PWQ)
    print("HMPVT PWQ is : ", HMPVT_PWQ)
    print("HPTF_RR PWQ is : ", HPTF_RR_PWQ)
    print("HMPVT_RR PWQ is : ", HMPVT_RR_PWQ)

    print("Optimal PWQ is : ", optimal_PWQ)
    print("ratio of optimal : ", ((optimal_PWQ - PPO_PWQ) / optimal_PWQ) * 100)
    max_PWQ = max(HPTF_RR_PWQ, HMPVT_RR_PWQ)
    max_PWQ = max(max_PWQ, HPTF_PWQ)
    max_PWQ = max(max_PWQ, HMPVT_PWQ)


    print("% : ", ((PPO_PWQ - max_PWQ) / PPO_PWQ) * 100)

    
    #print("version list : ", version_list)

    print("total episode cnt : ", episode_cnt)
    print("each episode's time : ", episode_time)
    #np.savetxt("transcoding2/arr.txt", version_list)

    PWVQ_values = [HPF_PWQ, HPTF_PWQ, HMPV_PWQ, HMPVT_PWQ, HPTF_RR_PWQ, HMPVT_RR_PWQ]
    average_PWQ = sum(PWVQ_values) / len(PWVQ_values)

    print("avg improvement % : ", ((PPO_PWQ - average_PWQ) / PPO_PWQ) * 100)

    '''
    for data in datasets:               # MCKP
        # 데이터셋에서 변수 초기화

        PWVQ = data['PWVQ_sum']
        energy = data['energy_sum']
        current_energy = energy

        numerator = data['PWVQ_arr'][:, 1:] - data['PWVQ_arr'][:, [0]]
        denominator = data['energy_arr'][:, 1:] - data['energy_arr'][:, [0]] 
        

        #numerator = data['PWVQ_arr'][:, 1:] - data['PWVQ_arr'][:, [0]]
        #denominator = data['energy_arr'][:, 1:] - data['energy_arr'][:, [0]]
        MCKP_2d_arr = np.zeros_like(denominator, dtype=float)
        np.divide(numerator, denominator, out=MCKP_2d_arr, where=denominator != 0)
        #print(MCKP_2d_arr)
        
        
        now_section_nst = np.zeros(N_section).astype(int)

        indices = np.array(np.unravel_index(np.argsort(-MCKP_2d_arr.ravel()), MCKP_2d_arr.shape)).T

        for section, nst in indices:
            print((section, nst))
            nst = nst + 1
            
            weight = (data['energy_arr'][section, nst] - data['energy_arr'][section, now_section_nst[section]])
            value = (data['PWVQ_arr'][section, nst] - data['PWVQ_arr'][section, now_section_nst[section]])

            #weight = (denominator[section, nst] - denominator[section, now_section_nst[section]])
            #value = (numerator[section, nst] - numerator[section, now_section_nst[section]])

            #if energy + weight <= ENERGY_LIMIT and nst > now_section_nst[section]:
            if energy + weight <= ENERGY_LIMIT:
                energy += weight
                PWVQ += value
                now_section_nst[section] = nst
                
        PWVQ_results.append(PWVQ)
    '''



def objective(trial):
    #gamma = trial.suggest_categorical('gamma', [0.989,0.99,0.991,0.992])
    #learning_rate = trial.suggest_categorical('learning_rate', [1e-4,2e-4,3e-4,4e-4])
    clip_range = trial.suggest_categorical('clip_range', [0.1,0.2,0.3,0.4])
    
    env: KnapsackEnv = or_gym.make("Knapsack-v0")  # , env_config=env_config)
    env = ActionMasker(env, mask_fn)
    #env.reset()

    model = MaskablePPO("MultiInputPolicy",
                         env,
                         clip_range=clip_range,
                         
                         tensorboard_log="./ppo_test",
                         verbose=1)
    
    model.learn(total_timesteps=50000)
    

    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    return mean_reward


def callback(study, trial, state={}):
    state.setdefault('trial_count', 0)
    state['trial_count'] += 1

    if state['trial_count'] >= 10 and trial.params['clip_range'] == 0.2 and trial.value == study.best_value:
        study.stop()