import pybullet as p
import time
import pybullet_data
import gym
import numpy as np
import time

from kinder_garten.envs.kinder_garten import KinderGarten

from stable_baselines3.common.callbacks import CheckpointCallback




# if __name__ == "__main__":
#     env = make_vec_env(lambda: KinderGarten("gripper",  'table', 'pybullet'), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
#     env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
#     model = SAC("CnnPolicy", env, buffer_size=10000, verbose=1, tensorboard_log=f'./{agent}/')
#     model.learn(total_timesteps=4e6, callback=checkpoint_callback)
#     model.save("ppo_cartpole")

env = KinderGarten("gripper",  'editor', 'pybullet', debug=True)

# env = KinderGarten("gripper",  'simple', 'pybullet')


idx = 0

while True:
    action = env.agent.action_space.sample()
    print(action)
    if type(env.agent.action_space) == gym.spaces.box.Box:
        
        # action = get_box(False)
        
        observation, reward, terminated, info = env.step(action)

        idx += 1
                
