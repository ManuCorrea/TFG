from kinder_garten.envs.kinder_garten import KinderGarten

from stable_baselines3 import PPO

# Save a checkpoint every 1000 steps
n_envs = 5
save_freq = 100000
save_freq = max(save_freq // n_envs, 1)

agent = "gripper"


env = KinderGarten("gripper",  'table', 'pybullet', debug=True)

# Here we load our desired model
model = PPO.load("depth2/CNN_rl_model_749970_steps.zip")
observation = env.reset()

while True:
    action, _states = model.predict(observation)
    
    observation, reward, terminated, info = env.step(action)

    print(reward)
                
