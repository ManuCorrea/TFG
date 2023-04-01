from kinder_garten.envs.kinder_garten import KinderGarten

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.logger import configure

ways = [DummyVecEnv, SubprocVecEnv, VecEnv]
ways_ = {DummyVecEnv:'DummyVecEnv', SubprocVecEnv:'SubprocVecEnv', VecEnv:'VecEnv'}

if __name__ == "__main__":
    for way in ways:
        for i in range(1, 8):
            new_logger = configure(f'perf/{ways_[way]}/{i}', ["stdout", "csv"])
            env = make_vec_env(lambda: KinderGarten("gripper",  'table', 'pybullet'), n_envs=i, vec_env_cls=way)
            model = PPO("MlpPolicy", env, verbose=1)
            model.set_logger(new_logger)
            model.learn(total_timesteps=20000)
            
