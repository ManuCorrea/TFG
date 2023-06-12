import time
import gym
import numpy as np
import time

from kinder_garten.envs.kinder_garten import KinderGarten


from xbox360controller import Xbox360Controller

controller = Xbox360Controller(axis_threshold=0.2)

env = KinderGarten("gripper",  'table', 'pybullet', debug=True)
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

    if show:
        print(data)
    
    return data

idx = 0
rewards = []
while True:
    action = env.agent.action_space.sample()
    if type(env.agent.action_space) == gym.spaces.box.Box:
        
        action = get_box(False)

        start = time.time()
        
        observation, reward, terminated, info = env.step(action)
        end = time.time() - start
        print(reward)
        rewards.append(reward)

        if terminated:
            print(f"reward: {sum(rewards)}")
            rewards = []

        idx += 1
                
