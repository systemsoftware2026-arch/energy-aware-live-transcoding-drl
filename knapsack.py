import numpy as np
import gym
from gym import spaces, logger
from gymnasium import spaces
from gym.utils import seeding
from scipy.stats import gamma
from or_gym.utils import assign_env_config
import copy
import random
import time
import math


np.random.seed(4)

#np.random.seed(10) # shuffle




GB = 1024               # MB is normal size
TB = 1024 * 1024
DAY = 24
FILE_LENGTH = 1200  # 1200 sec

VERSION = 7
#version_bitrate = np.array([30, 24, 15, 12, 10, 6, 4])      # Mbps
#version_bitrate = np.array([4, 6, 10, 12, 15, 24, 30])      # Mbps
version_bitrate = np.array([0.235, 0.375, 0.56, 1.05, 1.75, 2.35, 4.3])      # Mbps
version_bitrate_ratio = (version_bitrate / np.max(version_bitrate))
#MEAN_VMAF = np.array([43, 57.5, 67.5, 90.2, 93.8, 95.5, 100])
MEAN_VMAF = np.array([43, 57.5, 67.5, 75.2, 85.8, 92.5, 100])
MAX_VMAF = 100.1
DEVIATION = 4
SCAILING_FACTOR = 100

MAX_REQUEST_PER_HOUR = 10
#MAX_REQUEST_PER_HOUR = 5000

#zipf_parameter = 0.9
zipf_parameter = 0.729


alpha_beta = np.array([[0.1422, 1],       # a,b
                      [0.1468, 1],
                      [0.1592, 1],
                      [0.1719, 1],
                      [0.1836, 1],
                      [0.2167, 1],
                      [0.2803, 1]])




class KnapsackEnv(gym.Env):
    
    # Internal list of placed items for better rendering
    _collected_items = []
    
    def __init__(self, video_num = 300, *args, **kwargs):
        # Generate data with consistent random seed to ensure reproducibility
        #self.N = video_num            # number of video
        self.video_num = video_num  # 기본값 설정
        self.N = self.video_num  # number of videos로 할당

        self.max_weight = 1000000

        self.current_weight = 0
        self.mask = True
        self.seed = 0
        self.item_numbers = np.arange(self.N)
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
        self.randomize_params_on_reset = False
        self.item_limits = np.ones((VERSION - 1) * 2, dtype=np.int32)

        #self.N_scenario = 24
        self.N_scenario = 300
        self.group_size = 50
        self.N_group = int(self.N / self.group_size)
        self.N_group = math.ceil(self.N / self.group_size)
        self.current_PWQ = 0
        self.cumulative_transcoding_time = 0
        self.current_video = 0
        #self.noise_levels = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.noise_levels = np.linspace(10, 1, self.N_group)


        self.HPF_time = 0
        self.P_ij_time = 0
        self.P_ij_ratio_time = 0
        self.vmaf_time = 0
        self.vmaf_ratio_time = 0
        
        self.episode_cnt = 0
        self.start_time = 0
        self.end_time = 0
        self.cumulative_time = 0


        self._collected_items.clear()
        # Add env_config, if any
        assign_env_config(self, kwargs)
        self.set_seed()

        # init

        self.file_generate()
        
        self.video_popularity = self.generate_zipf_distribution(self.N, zipf_parameter)
        self.video_popularity = self.video_popularity * self.file_length
        self.video_popularity = self.video_popularity / np.sum(self.video_popularity)
        
        
        self.video_ranking = self.generate_ranking()
        
        
        self.rank_scenario = self.generate_add_noise()
        #self.rank_scenario = self.generate_shuffle_scenario()       # rank shuffle
        
        self.timeline_workload = self.generate_timeline()

        self.version_popularity_scenario = self.generate_version()
        self.initial_version_popularity = self.generate_version()
        self.version_popularity = self.initial_version_popularity.sum(axis = 0)
        self.version_popularity = self.version_popularity / np.sum(self.version_popularity)     # no avg but first value?

        self.timeline_sampled = np.random.choice(self.N_scenario, size = DAY, replace = False)      # sample 24 zipf timeline change ex) [2, 10, 5, ..., 77]
        

        vmaf = np.random.normal(MEAN_VMAF, DEVIATION, (self.N, VERSION))
        self.vmaf = np.clip(vmaf, None, 100)
        self.vmaf = np.sort(self.vmaf, axis = 1)
        self.vmaf[:, VERSION - 1] = 100

        self.zipf_value_scenario = self.generate_zipfvalue_scenario()

        
        #self.PWQ_threshold = (self.calculate_totalPWQ() - self.calculate_lowest_PWQ()) * 0.95
        self.PWQ_threshold = (self.calculate_totalPWQ() - self.calculate_lowest_PWQ()) * 0.93
        self.max_PWQ = (self.calculate_totalPWQ())
        self.max_value = (self.calculate_totalPWQ() - self.calculate_lowest_PWQ())
        
        
        self.transcoding_time = self.generate_transcoding_time()    # [scenario, video, ver] 3d_array
        self.avg_transcoding_time = np.mean(self.transcoding_time, axis = 0)
        self.transcoding_scenario = random.randint(0, self.N_scenario - 1)  # scenario (transcoding time)

        self.max_avg_transcoding_time = np.max(self.avg_transcoding_time)


        self.total_transcoding_time = np.sum(self.avg_transcoding_time[:, 1:]) # exclude lowest ver

        #self.total_transcoding_time = np.sum(self.avg_transcoding_time[:, :])  # include lowest ver

        #self.T_limit = np.sum(self.avg_transcoding_time[:, 0]) + ((np.sum(self.avg_transcoding_time[:, :]) - np.sum(self.avg_transcoding_time[:, 0])) * 0.4)


        #self.T_limit = self.total_transcoding_time * 0.25                # T_limit
        #self.T_limit = self.T_limit - np.sum(self.avg_transcoding_time[:, 0])
        self.T_limit = self.total_transcoding_time * 0.3
        print(self.T_limit)
        #exit()


        
        #print(self.HPF_alogorithm())
        #print(self.HPTF_alogorithm())
        #print(self.HPTF_RR_algorithm())
        #print("max is :", self.max_value)
        #exit()
        #value = (self.video_popularity[0] * self.version_popularity * self.vmaf[0] / self.avg_transcoding_time[0]) * SCAILING_FACTOR
        #print(value)
        #exit()


        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, VERSION), dtype=np.float32)
            
        #self.action_space = spaces.Discrete(self.N)
        self.action_space = spaces.MultiBinary(VERSION - 1)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=((VERSION - 1) * 2,), dtype=np.uint8),
                "avail_actions": spaces.Box(0, 1, shape=((VERSION - 1) * 2,), dtype=np.uint8),
                "state": obs_space
                })
        else:
            self.observation_space = spaces.Box(
                0, self.max_weight, shape=(2, self.N + 1), dtype=np.int32)
            

        #self.test()
        
        self.reset()



    def test(self):
        print("video popularity is : ", self.video_popularity)
        print("sum is : ", np.sum(self.video_popularity))
        print("ranking is : ", self.video_ranking)
        print("all version transcoding time is : ", np.sum(self.avg_transcoding_time))

        exit()

    
    def file_generate(self):
        #self.file_length = np.full(self.N, FILE_LENGTH)
        #self.file_length = np.random.randint(FILE_LENGTH - 100, FILE_LENGTH + 100, size=self.N)
        self.file_length = np.random.randint(FILE_LENGTH - 600, FILE_LENGTH + 600, size=self.N)
        self.file_bitrate = np.full(self.N, version_bitrate[VERSION - 1])                # version 7's bitrate (30Mbps)
        self.file_capacity = self.file_bitrate * self.file_length               # file storage is MB
        
        self.version_capacity = np.outer(self.file_capacity, version_bitrate_ratio)     # version's capacity (self.N, VERSION)
        
    
    
    
    def generate_zipf_distribution(self, n, a=1.0):
        ranks = np.arange(1, n + 1)
        weights = ranks ** (-a)
        return weights / weights.sum()

    def generate_ranking(self):
        sorted_indices = np.argsort(-self.video_popularity)
        ranking = np.zeros_like(sorted_indices)
        ranking[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        return ranking
    

    def generate_add_noise(self):
        rank_change_scenario = np.zeros((self.N_scenario, self.N)).astype(int)
        rank = self.video_ranking
        for scenario in range(self.N_scenario):
            noisy_rank = np.zeros(self.N)
            for i in range(self.N_group):
                start_idx = i * self.group_size
                
                #end_idx = start_idx + self.group_size
                #noise = np.random.normal(0, self.noise_levels[i], size = self.group_size)


                end_idx = min(start_idx + self.group_size, self.N)
                group_size_actual = end_idx - start_idx
                noise = np.random.normal(0, self.noise_levels[i], size=group_size_actual)

                
                noisy_rank[start_idx : end_idx] = rank[start_idx : end_idx] + noise
            sorted_idx = np.argsort(noisy_rank)
            unique_rank = np.zeros_like(rank)
            unique_rank[sorted_idx] = np.arange(1, self.N + 1)
            rank_change_scenario[scenario] = unique_rank

        #print("rank scenario is : ", rank_change_scenario[0])
        #exit()
        
        return rank_change_scenario
            

    def generate_shuffle_scenario(self):
        
        rank_change_scenario = np.zeros((self.N_scenario, self.N)).astype(int)
        video_num = np.argsort(self.video_ranking)
        arr = np.arange(1, self.N + 1)


        for scenario in range(self.N_scenario):
            
            shuffled_arr = np.zeros(self.N, dtype=int)
            for i in range(self.N_group):
                start_idx = i * self.group_size
                end_idx = min(start_idx + self.group_size, self.N)
                group_size_actual = end_idx - start_idx
                group = arr[start_idx:end_idx]
                np.random.shuffle(group)  # 그룹 내 셔플
                shuffled_arr[start_idx:end_idx] = group
            rank_change_scenario[scenario, video_num] = shuffled_arr
            


            #segmented_arr = np.split(arr, np.arange(self.group_size, self.N + 1, self.group_size))
            #for i in range(len(segmented_arr)):
                #np.random.shuffle(segmented_arr[i])
            #shuffled_arr = np.concatenate(segmented_arr)
            #rank_change_scenario[scenario, video_num] = shuffled_arr
        
        #print(rank_change_scenario[0])
        #exit()

        return rank_change_scenario



    

    def generate_version(self):
        SIZE = self.N
        version_scenario = np.zeros((self.N_scenario, SIZE))
        version_popularity = np.zeros((self.N_scenario, VERSION))
        
        mean = 3         # HVP : 4 MVP : 3 LVP : 2
        #mean = np.random.uniform(2, 4, size=self.N_scenario)
        #mean = np.random.uniform(0, 6, size=self.N_scenario)
        std_dev = 1.0
        mean_arr = np.random.normal(loc = mean, scale = std_dev, size = self.N_scenario)
        mean_arr[mean_arr > VERSION - 1] = VERSION - 1
        mean_arr[mean_arr < 0] = 0


        for i in range(self.N_scenario):
            now_mean = mean_arr[i]
            #now_std_dev = 1.5
            #now_std_dev = 2.5
            now_std_dev = 2.2  # MVP
            #now_std_dev = 1.7  # HVP
            version_scenario[i] = np.random.normal(loc = now_mean, scale = now_std_dev, size = SIZE)

        version_scenario[version_scenario > VERSION - 1] = VERSION - 1
        version_scenario[version_scenario < 0] = 0
        version_scenario = np.round(version_scenario).astype(int)

        for i in range(self.N_scenario):
            counts = np.zeros(VERSION).astype(int)
            for ver in version_scenario[i]:
                counts[ver] += 1
            version_popularity[i] = counts / SIZE
        
        #print(version_popularity)
        arr1d = np.mean(version_popularity, axis=0)
        print("aaa", arr1d)

        #exit()
        return version_popularity
    
    
    def generate_zipfvalue_scenario(self):
        scenario_zipfvalue = np.zeros((self.N_scenario, self.N))
        ranking_popularity = self.generate_zipf_distribution(self.N, zipf_parameter)
        for i in range(self.N_scenario):
            #parameter = random.uniform(0.7, 1.0)
            #ranking_popularity = self.generate_zipf_distribution(self.N, parameter)
            #ranking_popularity = self.generate_zipf_distribution(self.N, zipf_parameter)
            video_popularity = np.zeros(self.N)
            for idx, rank in enumerate(self.rank_scenario[i]):
                video_popularity[idx] = ranking_popularity[rank - 1] * self.file_length[idx]
            video_popularity = video_popularity / np.sum(video_popularity)
            scenario_zipfvalue[i] = video_popularity
        
        return scenario_zipfvalue

            


    def generate_timeline(self):
        DAY = 24
        split = int(DAY / 4)
        mean = 0.6              # base mean 0.6, std_dev 0.15
        std_dev = 0.2

        morning_workload = np.random.normal(loc = 0.23, scale = 0.05, size = split)
        afternoon_workload = np.random.normal(loc = 0.45, scale = 0.05, size = split)
        night_workload = np.random.normal(loc = 0.8, scale = 0.05, size = split)
        midnight_workload = np.random.normal(loc = 0.4, scale = 0.05, size = split)

        timeline_workload = np.concatenate((morning_workload, afternoon_workload, night_workload, midnight_workload))
        

        #timeline_workload = np.random.normal(loc = mean, scale = std_dev, size = DAY)
        #timeline_workload[timeline_workload < 0.1] = 0.1
        #timeline_workload[timeline_workload > 1] = 1

        timeline_request = (timeline_workload * MAX_REQUEST_PER_HOUR).astype(int)

        #print("workload : ", timeline_workload)
        #print("request : ", timeline_request)
        #exit()
        return timeline_request
    

    def calculate_totalPWQ(self):
        return_value = 0
        for video in range(self.N):
            version_QoE = 0
            for ver in range(VERSION):
                for hour in range(DAY):
                    version_QoE += self.zipf_value_scenario[self.timeline_sampled[hour], video] * self.version_popularity_scenario[self.timeline_sampled[hour], ver] * self.timeline_workload[hour] * self.vmaf[video, ver]
            return_value += version_QoE

        return return_value
    
    def calculate_PWQ(self, video, ver_list):                                       # normally calculate PWQ
        #check = np.array([1 if i in ver_list else 0 for i in range(VERSION)])
        check = np.array(ver_list)
        check = check[::-1]
        
        version_QoE = 0

        for ver in range(len(check)):
            if ver == VERSION - 1:
                continue
            if check[ver] == 0:
                degraded_ver = next((j for j in range(ver + 1, len(check)) if check[j] == 1), None)
                if degraded_ver is not None:
                    for hour in range(DAY):
                        version_QoE += self.zipf_value_scenario[self.timeline_sampled[hour], video] * self.version_popularity_scenario[self.timeline_sampled[hour], VERSION-1 - ver] * self.timeline_workload[hour] * self.vmaf[video, VERSION-1 - degraded_ver]
            else:
                for hour in range(DAY):
                    version_QoE += self.zipf_value_scenario[self.timeline_sampled[hour], video] * self.version_popularity_scenario[self.timeline_sampled[hour], VERSION-1 - ver] * self.timeline_workload[hour] * self.vmaf[video, VERSION-1 - ver]

        return version_QoE


    def calculate_lowest_PWQ(self):
        return_value = 0
        for video in range(self.N):
            version_QoE = 0
            for hour in range(DAY):
                version_QoE += self.zipf_value_scenario[self.timeline_sampled[hour], video] * self.version_popularity_scenario[self.timeline_sampled[hour], 0] * self.timeline_workload[hour] * self.vmaf[video, 0]
            return_value += version_QoE
        return return_value
        

    def generate_transcoding_time(self):
        k = 5.8
        theta = 3.4
        array_3d = np.zeros((self.N_scenario, self.N, VERSION))

        for scenario in range(self.N_scenario):
            for video in range(self.N):
                for ver in range(VERSION):
                    sample = gamma.rvs(k, scale = theta)
                    sample_adjusted = sample - k * theta
                    sample_adjusted = sample_adjusted / 100     # T_err
                    array_3d[scenario, video, ver] = alpha_beta[ver, 0] * (self.file_length[video] ** alpha_beta[ver, 1]) * (1 + sample_adjusted)
    
        return array_3d

        
    def get_current_PWQ(self):
        return self.current_PWQ
    
    def get_PWO_threshold(self):
        return self.PWQ_threshold
    
    def get_transcoding_time(self):
        return self.cumulative_transcoding_time
    
    def get_lowest_energy(self):
        lowest_energy = np.sum(self.transcoding_time[self.transcoding_scenario, :, 0])
        return lowest_energy



    def get_lowest_PWVQ(self):
        PWVQ = 0
        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            PWVQ += self.calculate_PWQ(video, check)
        return PWVQ
        
    


    

    def get_PPO_PWQ(self):
        return self.reward_sum
    
    def get_HPF_PWQ(self):
        return self.HPF_PWQ
    
    def get_HPTF_PWQ(self):
        return self.HPTF_PWQ
    
    def get_HMPV_PWQ(self):
        return self.HMPV_PWQ
    
    def get_HMPVT_PWQ(self):
        return self.HMPVT_PWQ
    
    def get_HPTF_RR_PWQ(self):
        return self.HPTF_RR_PWQ
    
    def get_HMPVT_RR_PWQ(self):
        return self.HMPVT_RR_PWQ
    
    def get_optimal_PWQ(self):
        return self.optimal_PWQ
    

    def get_episode_cnt(self):
        return self.episode_cnt
    
    def get_cumulative_time(self):
        return self.cumulative_time
    
    def get_video_num(self):
        return self.N
    

    def get_ver_list(self):
        return self.ver_list
    
    def comparison_process(self):
        self.HPF_PWQ = self.HPF_alogorithm()
        self.HPTF_PWQ = self.HPTF_alogorithm()
        self.HMPV_PWQ = self.HMPV_alogorithm()
        self.HMPVT_PWQ = self.HMPVT_alogorithm()
        self.HPTF_RR_PWQ = self.HPTF_RR_algorithm()
        self.HMPVT_RR_PWQ = self.HMPVT_RR_algorithm()
        self.optimal_PWQ = self.Optimal_algorithm()


    def set_E_limit(self, nst, number_of_nst):
        #number_of_nst = 10
        self.T_limit = self.total_transcoding_time * ((nst + 1) / number_of_nst)
        #self.T_limit = np.sum(self.avg_transcoding_time[:, 0]) + ((np.sum(self.avg_transcoding_time[:, :])) - np.sum(self.avg_transcoding_time[:, 0])) * ((nst + 1) / number_of_nst)


    def get_E_limit(self):
        return self.T_limit
    
    def get_cumulative_energy(self):
        return self.cumulative_transcoding_time
    
    def get_total_energy(self):
        return self.total_transcoding_time
    
    def get_HPF_cumulative_energy(self):
        return self.HPF_transcoding_time
    
    def get_HPTF_cumulative_energy(self):
        return self.HPTF_transcoding_time
    
    def get_HMPV_cumulative_energy(self):
        return self.HMPV_transcoding_time
    
    def get_HMPVT_cumulative_energy(self):
        return self.HMPVT_transcoding_time
    
    def get_HPTF_RR_cumulative_energy(self):
        return self.HPTF_RR_transcoding_time
    
    def get_HMPVT_RR_cumulative_energy(self):
        return self.HMPVT_RR_transcoding_time
    
        
    

    def Optimal_algorithm(self):

        cumulative_QoE = 0
        check_transcoding = np.ones(VERSION)
        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, check_transcoding)
            cumulative_QoE += (now_PWQ - prev_PWQ)

        return cumulative_QoE


    def HPF_alogorithm(self):
        transcoding_time = 0
        cumulative_QoE = 0

        check_transcoding = np.zeros((self.N, VERSION))
        check_transcoding[:, 0] = 1
        
        P_ij = np.outer(self.video_popularity, self.version_popularity)
        
        flattened_array = P_ij.flatten()
        sorted_idx = np.argsort(flattened_array)[::-1]

        for idx in sorted_idx:
            
            current_video, current_version = divmod(idx, P_ij.shape[1])
            if current_version == 0: continue   # except lowest version
            #print("idx : ", current_version)
            if transcoding_time + self.transcoding_time[self.transcoding_scenario, current_video, current_version] > self.T_limit:
                break
            transcoding_time += self.transcoding_time[self.transcoding_scenario, current_video, current_version]
            check_transcoding[current_video, current_version] = 1


        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, check_transcoding[video])
            cumulative_QoE += (now_PWQ - prev_PWQ)

        
        self.HPF_transcoding_time = transcoding_time
        #print(check_transcoding)

        return cumulative_QoE
    

    def HPTF_alogorithm(self):
        transcoding_time = 0
        cumulative_QoE = 0

        check_transcoding = np.zeros((self.N, VERSION))
        check_transcoding[:, 0] = 1
        
        P_ij = np.outer(self.video_popularity, self.version_popularity)
        divided_arr = np.divide(P_ij, self.avg_transcoding_time)
        
        flattened_array = divided_arr.flatten()
        sorted_idx = np.argsort(flattened_array)[::-1]
        

        for idx in sorted_idx:
            current_video, current_version = divmod(idx, divided_arr.shape[1])
            if current_version == 0: continue   # except lowest version
            #print("idx : ", current_version)
            if transcoding_time + self.transcoding_time[self.transcoding_scenario, current_video, current_version] > self.T_limit:
                break
            transcoding_time += self.transcoding_time[self.transcoding_scenario, current_video, current_version]
            check_transcoding[current_video, current_version] = 1


        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, check_transcoding[video])
            cumulative_QoE += (now_PWQ - prev_PWQ)
        
        #print(check_transcoding)
        self.HPTF_transcoding_time = transcoding_time

        return cumulative_QoE
    

    def HMPV_alogorithm(self):
        transcoding_time = 0
        cumulative_QoE = 0

        check_transcoding = np.zeros((self.N, VERSION))
        check_transcoding[:, 0] = 1
        
        P_ij = np.outer(self.video_popularity, self.version_popularity)
        P_ij = P_ij * self.vmaf
        
        flattened_array = P_ij.flatten()
        sorted_idx = np.argsort(flattened_array)[::-1]

        for idx in sorted_idx:
            
            current_video, current_version = divmod(idx, P_ij.shape[1])
            if current_version == 0: continue   # except lowest version
            #print("idx : ", current_version)
            if transcoding_time + self.transcoding_time[self.transcoding_scenario, current_video, current_version] > self.T_limit:
                break
            transcoding_time += self.transcoding_time[self.transcoding_scenario, current_video, current_version]
            check_transcoding[current_video, current_version] = 1


        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, check_transcoding[video])
            cumulative_QoE += (now_PWQ - prev_PWQ)

        #print(check_transcoding)
        self.HMPV_transcoding_time = transcoding_time

        return cumulative_QoE
    


    def HMPVT_alogorithm(self):
        transcoding_time = 0
        cumulative_QoE = 0

        check_transcoding = np.zeros((self.N, VERSION))
        check_transcoding[:, 0] = 1
        
        P_ij = np.outer(self.video_popularity, self.version_popularity)
        P_ij = P_ij * self.vmaf
        divided_arr = np.divide(P_ij, self.avg_transcoding_time)
        
        flattened_array = divided_arr.flatten()
        sorted_idx = np.argsort(flattened_array)[::-1]

        for idx in sorted_idx:
            
            current_video, current_version = divmod(idx, divided_arr.shape[1])
            if current_version == 0: continue   # except lowest version
            #print("idx : ", current_version)
            if transcoding_time + self.transcoding_time[self.transcoding_scenario, current_video, current_version] > self.T_limit:
                break
            transcoding_time += self.transcoding_time[self.transcoding_scenario, current_video, current_version]
            check_transcoding[current_video, current_version] = 1


        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, check_transcoding[video])
            cumulative_QoE += (now_PWQ - prev_PWQ)

        #print(check_transcoding)
        self.HMPVT_transcoding_time = transcoding_time

        return cumulative_QoE



    

    def HPTF_RR_algorithm(self):
        transcoding_time = 0
        cumulative_QoE = 0
        cur_video = 0
        transcoded_check = np.zeros((self.N, VERSION))
        transcoded_check[:, 0] = 1

        cnt = 0

        while True:
            P_ij = self.video_popularity[cur_video] * self.version_popularity
            divided_arr = P_ij / self.avg_transcoding_time[cur_video]
            sorted_idx = np.argsort(divided_arr)[::-1]
            
            for idx in sorted_idx:
                if transcoded_check[cur_video, idx] == 0:
                    transcoded_check[cur_video, idx] = 1
                    #print("idx : ", idx)
                    transcoding_time += self.transcoding_time[self.transcoding_scenario, cur_video, idx]
                    break
            if transcoding_time > self.T_limit:
                break
            
            if cur_video == self.N - 1:
                cur_video = 0
                cnt += 1
            else:
                cur_video += 1
        
        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, transcoded_check[video])
            cumulative_QoE += (now_PWQ - prev_PWQ)
            
        #print(transcoded_check)
        #print("cnt is ", cnt)
        self.HPTF_RR_transcoding_time = transcoding_time
    
        return cumulative_QoE
    


    def HMPVT_RR_algorithm(self):
        transcoding_time = 0
        cumulative_QoE = 0
        cur_video = 0
        transcoded_check = np.zeros((self.N, VERSION))
        transcoded_check[:, 0] = 1

        cnt = 0

        while True:
            P_ij = self.video_popularity[cur_video] * self.version_popularity
            P_ij = P_ij * self.vmaf[cur_video]
            divided_arr = P_ij / self.avg_transcoding_time[cur_video]
            sorted_idx = np.argsort(divided_arr)[::-1]
            
            for idx in sorted_idx:
                if transcoded_check[cur_video, idx] == 0:
                    transcoded_check[cur_video, idx] = 1
                    #print("idx : ", idx)
                    transcoding_time += self.transcoding_time[self.transcoding_scenario, cur_video, idx]
                    break
            if transcoding_time > self.T_limit:
                break
            
            if cur_video == self.N - 1:
                cur_video = 0
                cnt += 1
            else:
                cur_video += 1
        
        check = np.zeros(VERSION)
        check[0] = 1
        for video in range(self.N):
            prev_PWQ = self.calculate_PWQ(video, check)
            now_PWQ = self.calculate_PWQ(video, transcoded_check[video])
            cumulative_QoE += (now_PWQ - prev_PWQ)
            
        #print(transcoded_check)
        #print("cnt is ", cnt)
        self.HMPVT_RR_transcoding_time = transcoding_time
    
        return cumulative_QoE


    
    

    def _STEP(self, item):
        # Check that item will fit
        #print("action is : ", item)     # item is numpy array ex) [0 0 1 1 0 1]
        truncated = False
        reward = 0
        self.ver_list[self.current_video, 1:] = item

        tmp_version = np.zeros(VERSION)
        tmp_version[1:] = item
        indices = np.where(tmp_version == 1)[0]
        transcoing_time = np.sum(self.transcoding_time[self.transcoding_scenario, self.current_video, indices])
        # calculate sum of current video's transcoding time
        prev_PWQ = 0

        if self.cumulative_transcoding_time + transcoing_time <= self.T_limit:
            self.cumulative_transcoding_time += transcoing_time
            self.check_version[self.current_video, 1:] = item
            video_PWQ = 0
            prev_check = np.zeros(VERSION)
            prev_check[0] = 1
            prev_PWQ = self.calculate_PWQ(self.current_video, prev_check)
            now_PWQ = self.calculate_PWQ(self.current_video, self.check_version[self.current_video])
            video_PWQ = (now_PWQ - prev_PWQ)
            reward = video_PWQ
            #print("reward : ", reward)
        
            done = False
        
        else:
            reward = 0
            done = True
            print("cur video num is : ", self.current_video)
                
                
            HPTF_RR_value = self.HPTF_RR_algorithm()
            HMPVT_RR_value = self.HMPVT_RR_algorithm()
            print("                 PPO's PWQ is : ", self.reward_sum)
            print("!!!!!HPTF's PWQ is : ", self.HPTF_alogorithm())
            print("!!HPTF_RR's PWQ is : ", HPTF_RR_value)
            print("!!!!!HMPVT_RR's PWQ is : ", HMPVT_RR_value)
            max_value = max(HPTF_RR_value, HMPVT_RR_value)
            print("% : ", ((self.reward_sum - max_value) / max_value) * 100)
            #print("total PWQ is : ", self.max_PWQ - self.calculate_lowest_PWQ())
            self.end_time = time.time()
            self.cumulative_time += (self.end_time - self.start_time)
            
        
        self.reward_sum += reward            



        
        
        if self.current_video == self.N - 1:
            HPTF_RR_value = self.HPTF_RR_algorithm()
            HMPVT_RR_value = self.HMPVT_RR_algorithm()
            print("??????")
            print("                 PPO's PWQ is : ", self.reward_sum)
            print("!!!!!HPTF's PWQ is : ", self.HPTF_alogorithm())
            print("!!HPTF_RR's PWQ is : ", HPTF_RR_value)
            print("!!!!!HMPVT_RR's PWQ is : ", HMPVT_RR_value)
            max_value = max(HPTF_RR_value, HMPVT_RR_value)
            print("% : ", ((self.reward_sum - max_value) / max_value) * 100)

            
            done = True
            self.end_time = time.time()
            self.cumulative_time += (self.end_time - self.start_time)

        

        if self.current_video < self.N - 1:
            self.current_video += 1
        

        self._update_state(item)


        return self.state, reward,  done, truncated, {'action_mask': self.item_limits}


        


    
    def _get_obs(self):
        return self.state
    
    def _update_state(self, item=None):

        #if item is not None:
            #self.item_limits[item] -= 1
        
        #if self.current_video >= self.N - 2:
            #return
        if self.current_video >= self.N - 4:
            return
        '''
        # state 1개
        state_items = np.vstack([
            ((self.video_popularity[self.current_video] * self.version_popularity[1:] * self.vmaf[self.current_video, 1:] / self.avg_transcoding_time[self.current_video, 1:]) * SCAILING_FACTOR)
        ], dtype = np.float32)
        '''

        
        # state 3개
        state_items = np.vstack([
            ((self.video_popularity[self.current_video] * self.version_popularity[1:] * self.vmaf[self.current_video, 1:] / self.avg_transcoding_time[self.current_video, 1:]) * SCAILING_FACTOR),
            ((self.video_popularity[self.current_video + 1] * self.version_popularity[1:] * self.vmaf[self.current_video + 1, 1:] / self.avg_transcoding_time[self.current_video + 1, 1:]) * SCAILING_FACTOR),
            ((self.video_popularity[self.current_video + 2] * self.version_popularity[1:] * self.vmaf[self.current_video + 2, 1:] / self.avg_transcoding_time[self.current_video + 2, 1:]) * SCAILING_FACTOR)
        ], dtype = np.float32)
        

        
        '''
        # state 1개
        state = np.hstack([
            state_items,
            np.array([[self.current_video / self.N]
                      ])
        ])
        '''
        
        
        # state 3개
        state = np.hstack([
            state_items,
            np.array([[self.current_video / self.N],
                      [self.cumulative_transcoding_time / self.T_limit],
                      [self.reward_sum / self.max_PWQ]
                      
                      ])
        ])
        

        if self.mask:
            #mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
            mask = np.zeros((VERSION - 1) * 2)
            
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones((VERSION - 1) * 2, dtype=np.uint8),
                "state": state
                }
        else:
            state = np.vstack([
                self.item_weights,
                self.item_values], dtype=np.int32)
            self.state = np.hstack([
                state,
                np.array([
                    [self.max_weight],
                     [self.current_weight]])
                ])        
            

    def decision_reset(self):
        self.reward_sum = 0
        self.cumulative_transcoding_time = 0
        self.current_video = 0
        
        

        self.HPF_transcoding_time = 0
        self.HPTF_transcoding_time = 0
        self.HMPV_transcoding_time = 0
        self.HMPVT_transcoding_time = 0
        self.HPTF_RR_transcoding_time = 0
        self.HMPVT_RR_transcoding_time = 0
    
    def _RESET(self):


        self.start_time = time.time()
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self.reward_sum = 0
        self.cumulative_transcoding_time = 0

        self.episode_cnt += 1
        
        

        self.item_limits = np.ones((VERSION - 1) * 2, dtype=np.int32)
        self.timeline_sampled = np.random.choice(self.N_scenario, size = DAY, replace = False)      # sample 24 zipf timeline change ex) [2, 10, 5, ..., 77]
        #self.PWQ_threshold = (self.calculate_totalPWQ() - self.calculate_lowest_PWQ()) * 0.95
        self.PWQ_threshold = (self.calculate_totalPWQ() - self.calculate_lowest_PWQ()) * 0.93
        self.max_PWQ = (self.calculate_totalPWQ())
        
        self.check_version = np.zeros((self.N, VERSION))
        self.check_version[:, 0] = 1


        self.current_video = 0


        self.current_PWQ = 0        # current weight





        self.transcoding_scenario = random.randint(0, self.N_scenario - 1)
        #self.timeline_workload = self.generate_timeline()
        self.ver_list = np.zeros((self.N, VERSION))

        self._collected_items.clear()
        self._update_state()

        return self.state, {'action_mask': self.item_limits}
    
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def valid_action_mask(self):
        return self.item_limits

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)
        
    def render(self):
        total_value = 0
        total_weight = 0
        for i in range(self.N) :
            if i in self._collected_items :
                total_value += self.item_values[i]
                total_weight += self.item_weights[i]
        print(self._collected_items, total_value, total_weight)
        
        # RlLib requirement: Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.
        return True























































































class BinaryKnapsackEnv(KnapsackEnv):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
        assign_env_config(self, kwargs)

        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(len(self.item_limits),), dtype=np.uint8),
                "avail_actions": spaces.Box(0, 1, shape=(len(self.item_limits),), dtype=np.uint8),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space

        self.reset()

    def _STEP(self, item):
        # Check item limit
        if self.item_limits[item] > 0:
            # Check that item will fit
            if self.item_weights[item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[item]
                reward = self.item_values[item]
                if self.current_weight == self.max_weight:
                    done = True
                else:
                    done = False
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True
            
        return self.state, reward, done, {}

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ], dtype=np.int32)
        state = np.hstack([
            state_items, 
            np.array([[self.max_weight],
                      [self.current_weight], 
                      [0] # Serves as place holder
                ])
        ], dtype=np.int32)
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
            mask = np.where(self.item_limits > 0, mask, 0)
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N, dtype=np.uint8),
                "state": state
            }
        else:
            self.state = state.copy()
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self.item_limits = np.ones(self.N, dtype=np.int32)
        self._update_state()
        return self.state

class BoundedKnapsackEnv(KnapsackEnv):
    
    def __init__(self, *args, **kwargs):
        self.N = 200
        self.item_limits_init = np.random.randint(1, 10, size=self.N, dtype=np.int32)
        self.item_limits = self.item_limits_init.copy()
        super().__init__()
        self.item_weights = np.random.randint(1, 100, size=self.N, dtype=np.int32)
        self.item_values = np.random.randint(0, 100, size=self.N, dtype=np.int32)

        assign_env_config(self, kwargs)

        obs_space = spaces.Box(
            0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(len(self.item_limits),), dtype=np.uint8),
                "avail_actions": spaces.Box(0, 1, shape=(len(self.item_limits),), dtype=np.uint8),
                "state": obs_space
            })
        else:
            self.observation_space = obs_space
        
    def _STEP(self, item):
        # Check item limit
        if self.item_limits[item] > 0:
            # Check that item will fit
            if self.item_weights[item] + self.current_weight <= self.max_weight:
                self.current_weight += self.item_weights[item]
                reward = self.item_values[item]
                if self.current_weight == self.max_weight:
                    done = True
                else:
                    done = False
                self._update_state(item)
            else:
                # End if over weight
                reward = 0
                done = True
        else:
            # End if item is unavailable
            reward = 0
            done = True
            
        return self.state, reward, done, {}

    def _update_state(self, item=None):
        if item is not None:
            self.item_limits[item] -= 1
        state_items = np.vstack([
            self.item_weights,
            self.item_values,
            self.item_limits
        ], dtype=np.int32)
        state = np.hstack([
            state_items, 
            np.array([[self.max_weight],
                      [self.current_weight], 
                      [0] # Serves as place holder
                ], dtype=np.int32)
        ])
        if self.mask:
            mask = np.where(self.current_weight + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
            mask = np.where(self.item_limits > 0, mask, 0)
            self.state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.N, dtype=np.uint8),
                "state": state
            }
        else:
            self.state = state.copy()
        
    def sample_action(self):
        return np.random.choice(
            self.item_numbers[np.where(self.item_limits!=0)])
    
    def _RESET(self):
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N, dtype=np.int32)
            self.item_values = np.random.randint(0, 100, size=self.N, dtype=np.int32)
            self.item_limits = np.random.randint(1, 10, size=self.N, dtype=np.int32)
        else:
            self.item_limits = self.item_limits_init.copy()

        self.current_weight = 0
        self._update_state()
        return self.state

