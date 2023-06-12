from kinder_garten.envs.kinder_garten import KinderGarten

env = KinderGarten(agent="gripper", scene_name='table', engine='pybullet', debug=True)

while True:
    action = env.agent.action_space.sample()     
    observation, reward, terminated, info = env.step(action)
    