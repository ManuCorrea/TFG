import gym

from kinder_garten.envs.kinder_garten import KinderGarten

env = KinderGarten("gripper",  'editor', 'pybullet', debug=True)

# env = KinderGarten("gripper",  'simple', 'pybullet')


idx = 0

action = env.agent.action_space.sample()


while True:    
    if type(env.agent.action_space) == gym.spaces.box.Box:
        
        observation, reward, terminated, info = env.step([0, 0, 0, 0, 0])

        idx += 1
        print(f"idx: {idx}")
        if idx == 25:
            print(action)
            env.reset()
            print("resetting")
            idx = 0
                
