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

# Training

# Save a checkpoint every 1000 steps
n_envs = 6
save_freq = 100000
save_freq = max(save_freq // n_envs, 1)

checkpoint_callback = CheckpointCallback(
  save_freq=save_freq,
  save_path="./logsEvitaPellizcos/",
  name_prefix="CNN_rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)



# a =  KinderGarten("gripper",  'table', 'pybullet')
# check_env(a)
# Parallel environments

# 3 envs 24fps

# TODO check camera output

agent = "gripper"

if __name__ == "__main__":
    env = make_vec_env(lambda: KinderGarten("gripper",  'table', 'pybullet'), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)

    # model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f'./{agent}/') # buffer_size=100000,
    model = PPO.load("logs/CNN_rl_model_899964_steps.zip",
                     print_system_info=True, env=env)

    model.learn(total_timesteps=4e6, callback=checkpoint_callback)
    model.save("final")

# env = KinderGarten("gripper",  'table', 'pybullet', debug=True)

# env = KinderGarten("gripper",  'simple', 'pybullet')


# from xbox360controller import Xbox360Controller

# controller = Xbox360Controller(axis_threshold=0.2)

def get_box(show=True):
    data = np.zeros(5, dtype=np.float64)
    lx = controller.axis_l.x
    ly = controller.axis_l.y
    if abs(lx) > abs(ly):
        data[0] = lx
    else:
        data[1] = ly

    rx = controller.axis_r.x
    ry = controller.axis_r.y

    if abs(rx) > abs(ry):
        data[3] = -1*rx
    else:
        data[2] = ry

    tl = controller.trigger_l._value
    tr = controller.trigger_r._value
    if abs(tl) > abs(tr):
        data[4] = -tl
    else:
        data[4] = tr
    data = np.around(data, 1)
    # badIndices = (data < 0.2 | data > )
    # data[badIndices] = 0

    if show:
        print(data)
    
    return data

# idx = 0

# while True:
#     action = env.agent.action_space.sample()
#     print(action)
#     if type(env.agent.action_space) == gym.spaces.box.Box:
        
#         action = get_box(False)

#         # if idx == 100:
#         #     cProfile.run('env.step(action)', filename='env.dat')
        
#         start = time.time()
        
#         observation, reward, terminated, info = env.step(action)
#         end = time.time() - start

#         print(f'Took {end}')
#         idx += 1
                
