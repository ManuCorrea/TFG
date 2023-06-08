import pybullet as p
import time
import pybullet_data
import gym
import numpy as np
import time
import time

from kinder_garten.envs.kinder_garten import KinderGarten


env = KinderGarten("gripper",  'table', 'pybullet', debug=True)
# env = KinderGarten("gripper",  'simple', 'pybullet')

idx = 0
rewards = []
while True:
    action = env.agent.action_space.sample()
    if type(env.agent.action_space) == gym.spaces.box.Box:
        
        action = env.action_space.sample()

        # if idx == 100:
        #     cProfile.run('env.step(action)', filename='env.dat')
        
        start = time.time()
        
        observation, reward, terminated, info = env.step(action)
        end = time.time() - start
        rewards.append(reward)

        if terminated:
            print(f"reward: {sum(rewards)}")
            rewards = []

        idx += 1
                
