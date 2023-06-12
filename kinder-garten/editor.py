import gym

from kinder_garten.envs.kinder_garten import KinderGarten

env = KinderGarten("gripper",  'editor', 'pybullet', debug=True)

# env = KinderGarten("gripper",  'simple', 'pybullet')


idx = 0

action = env.agent.action_space.sample()


while True:    
    observation, reward, terminated, info = env.step([0, 0, 0, 0, 0])

    if idx == 25:
        env.reset()
                
