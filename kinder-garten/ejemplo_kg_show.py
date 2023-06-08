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

# shows the results of a trained model

# Save a checkpoint every 1000 steps
n_envs = 5
save_freq = 100000
save_freq = max(save_freq // n_envs, 1)

# a =  KinderGarten("gripper",  'table', 'pybullet')
# check_env(a)
# Parallel environments

# 3 envs 24fps

# TODO check camera output

agent = "gripper"

# if __name__ == "__main__":
#     env = make_vec_env(lambda: KinderGarten("gripper",  'table', 'pybullet'), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
#     env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
#     model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f'./{agent}/') # buffer_size=100000,
#     model.learn(total_timesteps=4e6)
#     model.save("final")

env = KinderGarten("gripper",  'table', 'pybullet', debug=True)

# env = KinderGarten("gripper",  'simple', 'pybullet')


# from xbox360controller import Xbox360Controller

# controller = Xbox360Controller(axis_threshold=0.2)


idx = 0

model = PPO.load("logs/CNN_rl_model_899964_steps.zip")
observation = env.reset()

while True:
    # action = env.agent.action_space.sample()
    # print(action)
    if type(env.agent.action_space) == gym.spaces.box.Box:
        
        # action = get_box(False)

        action, _states = model.predict(observation)

        # if idx == 100:
        #     cProfile.run('env.step(action)', filename='env.dat')
        
        start = time.time()
        
        observation, reward, terminated, info = env.step(action)
        print(reward)
        end = time.time() - start

        # print(f'Took {end}')
        idx += 1
                
