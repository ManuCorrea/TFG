import pybullet as p
import numpy as np
import time

from kinder_garten.envs.kinder_garten import KinderGarten

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from stable_baselines3.common.callbacks import CheckpointCallback

# Training
n_envs = 6
save_freq = 50000
save_freq = max(save_freq // n_envs, 1)

checkpoint_callback = CheckpointCallback(
  save_freq=save_freq,
  save_path="./depth2/",
  name_prefix="CNN_rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

agent = "gripper"

if __name__ == "__main__":
    env = make_vec_env(lambda: KinderGarten("gripper",  'table', 'pybullet'), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f'./{agent}/') # buffer_size=100000,
    # model = PPO.load("logs/CNN_rl_model_899964_steps.zip",
    #                  print_system_info=True, env=env)

    model.learn(total_timesteps=4e6, callback=checkpoint_callback)
    model.save("final")

