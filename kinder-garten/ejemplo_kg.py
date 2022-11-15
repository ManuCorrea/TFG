import pybullet as p
import time
import pybullet_data

from kinder_garten.envs.kinder_garten import KinderGarten

env = KinderGarten("gripper",  'table', 'pybullet')
# env = KinderGarten("gripper",  'simple', 'pybullet')

while True:
    action = env.agent.action_space.sample()

    observation, reward, terminated, _, info = env.step(action)
