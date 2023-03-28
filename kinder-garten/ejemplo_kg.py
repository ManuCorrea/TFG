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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_checker import check_env
import cProfile

# from xbox360controller import Xbox360Controller

# controller = Xbox360Controller(axis_threshold=0.2)

# a =  KinderGarten("gripper",  'table', 'pybullet')
# check_env(a)
# Parallel environments

# TODO use monitor https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
# TODO use callback, keep best models

if __name__ == "__main__":
    env = make_vec_env(lambda: KinderGarten("gripper",  'table', 'pybullet'), n_envs=3, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=4e6)
    model.save("ppo_cartpole")

# env = KinderGarten("gripper",  'table', 'pybullet')

# env = KinderGarten("gripper",  'simple', 'pybullet')


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
#     # action = env.agent.action_space.sample()
#     if type(env.agent.action_space) == gym.spaces.box.Box:
        
#         action = get_box(False)

#         # if idx == 100:
#         #     cProfile.run('env.step(action)', filename='env.dat')
        
#         start = time.time()
        
#         observation, reward, terminated, _, info = env.step(action)
#         end = time.time() - start

#         print(f'Took {end}')
#         idx += 1
                
