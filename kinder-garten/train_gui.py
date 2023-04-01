import tkinter as tk
from tkinter import ttk
from tkinter import EventType
import math
from stable_baselines3 import A2C, DDPG, DQN, HerReplayBuffer, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# genera ventana de TK, ver pq
# from kinder_garten.envs.kinder_garten import KinderGarten

# Import module
from tkinter import *
  
  
RL_ALGORITHMS_FUNCTIONS = {
    "A2C":A2C,
    "DDPG":DDPG,
    "DQN":DQN,
    "HER":HerReplayBuffer,
    "PPO":PPO,
    "SAC":SAC,
    "TD3":TD3
}

AGENTS_GYM = ["CartPole-v1"]

AGENTS_KG = ["gripper", "Walker"]
SCENES_KG = ["Plane", "table"]

# Change the label text
def show():
    rl_alg = rl_algo_menu.get()
    agent = agent_menu.get()
    time_steps = int(time_steps_input.get())
    if agent in AGENTS_GYM:
        env = make_vec_env("CartPole-v1", n_envs=2)
    elif agent in AGENTS_KG:
        wrappable_env = KinderGarten(agent,  scene, 'pybullet')
        env = make_vec_env(lambda: wrappable_env, n_envs=3, vec_env_cls=SubprocVecEnv)

    scene = scene_menu.get()
    root.destroy()
    rl_func = RL_ALGORITHMS_FUNCTIONS[rl_alg]
    
    
    model = rl_func("MlpPolicy", env, device="cpu")
    model.learn(total_timesteps=25000)
  
# Dropdown menu options
RL_ALGORITHMS = [
    "A2C",
    "DDPG",
    "DQN",
    "HER",
    "PPO",
    "SAC",
    "TD3"
]

AGENT = [
    "Gripper",
    "Walker",
    "CartPole-v1"
]

SCENE = [
    "Plane",
    "Table"
]

POLICY = [
    "CnnPolicies",
    "MlpPolicies"
]
  
if __name__ == "__main__":
    # Create object
    root = Tk()
    
    # Adjust size
    root.geometry( "200x200" )

    rl_algo_menu = StringVar()
    rl_algo_menu.set("A2C")
    drop = OptionMenu( root , rl_algo_menu , *RL_ALGORITHMS )
    drop.grid()

    agent_menu = StringVar()
    agent_menu.set("Gripper")
    drop = OptionMenu( root , agent_menu , *AGENT )
    drop.grid()

    scene_menu = StringVar()
    scene_menu.set("Plane")
    drop = OptionMenu( root , scene_menu , *SCENE )
    drop.grid()
    
    # Create button, it will change label text
    button = Button( root , text = "Train" , command = show ).grid()
    
    time_steps_input = Entry(root)
    time_steps_input.insert(0, "100000")
    time_steps_input.grid()

    # Create Label
    label = Label( root , text = " " )
    label.grid()

    tk.Label(root, text="time_steps").grid(row=5)
    tk.Label(root, text="Last Name").grid(row=6)

    e1 = tk.Entry(root)
    e2 = tk.Entry(root)
    e1.insert(10, "100000")
    e2.insert(10, "Jill")

    e1.grid(row=5, column=1)
    e2.grid(row=6, column=1)

    # Execute tkinter
    root.mainloop()