import pybullet as p
import time
import gym
import time
import cv2
from kinder_garten.envs.kinder_garten import KinderGarten
from stable_baselines3.common.env_checker import check_env



env = KinderGarten(agent="gripper", scene_name='table',
                   engine='pybullet')


env = KinderGarten(agent="gripper", scene_name='table', engine='pybullet', debug=True)
check_env(env, warn=True)

# env = KinderGarten("gripper",  'simple', 'pybullet')

idx = 0
rewards = []
while True:
    action = env.agent.action_space.sample()
    if type(env.agent.action_space) == gym.spaces.box.Box:
        
        action = env.action_space.sample()
        
        start = time.time()
        
        observation, reward, terminated, info = env.step(action)
        end = time.time() - start
        rewards.append(reward)

        cv2.imshow('Frame', observation[0])

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


        if terminated:
            print(f"reward: {sum(rewards)}")
            rewards = []

        idx += 1
                
