import pybullet as p
import time
import pybullet_data
import gym
import numpy as np
import time
from threading import Thread
import time

from kinder_garten.envs.kinder_garten import KinderGarten

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import cProfile

from stable_baselines3.common.callbacks import CheckpointCallback


# show trained model

# shows the results of a trained model

# Save a checkpoint every 1000 steps
n_envs = 5
save_freq = 100000
save_freq = max(save_freq // n_envs, 1)

agent = "gripper"


env = KinderGarten("gripper",  'table', 'pybullet', debug=True)


idx = 0

# funciona un poco
# model = PPO.load("depth2/CNN_rl_model_349986_steps.zip")
model = PPO.load("depth2/CNN_rl_model_749970_steps.zip")
observation = env.reset()

while True:
    action, _states = model.predict(observation)
    
    start = time.time()
    
    observation, reward, terminated, info = env.step(action)
    end = time.time() - start

    print(reward)
    idx += 1
                
