import logging
import gym

from kinder_garten.envs.scene import scene
from kinder_garten.envs.agents import gripper
from pybullet_utils import bullet_client
import pybullet as p

logging.basicConfig(filename='gripper.log', level=logging.INFO,
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S')


# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
RED = "\x1b[31;20m"
RESET = "\x1b[0m"

logFormatter = logging.Formatter(
    RED + "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s" + RESET)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logging.getLogger().addHandler(consoleHandler)


class KinderGarten(gym.Env):

    def __init__(self, agent, scene_name, engine, debug=False):

        self.engine = engine

        self.agent = None
        self.world = None  # pass engine?
        self.reward = None  # pass all from before
        self.debug = debug

        if self.engine == 'pybullet':
            if self.debug:
                self.physicsClient = bullet_client.BulletClient(p.GUI)
                self.prev_reset_btn = 0
                self.reset_btn = self.physicsClient.addUserDebugParameter('reset', 1,
                                                 0, 0)
            else:
                self.physicsClient = bullet_client.BulletClient(p.DIRECT)
            p.setGravity(0, 0, -10)

        # create different worlds
        # which will be passed to the agent
        # which will be passed to the rewards

        self.scene_editor = False
        if scene_name == 'simple':
            self.scene = scene.Scene(self.engine, scene_name,  self.physicsClient)
        elif scene_name == 'table':
            self.scene = scene.Scene(self.engine, scene_name,  self.physicsClient)
        elif scene_name == 'editor':
            from kinder_garten.GUI.utils import UserInterface
            self.user_interface = UserInterface()
            self.scene_editor = True
            self.scene = scene.Scene(self.engine, scene_name,  self.physicsClient)
        else:
            # TODO verify custom scene object
            pass
        

        if agent == 'gripper':
            self.agent = gripper.Gripper(self.engine, self.physicsClient, self.reward, debug=True)
        else:
            # load custom class that expects certain format?
            pass

        self.observation_space = self.agent.observation_space
        self.action_space = self.agent.action_space


    def reset(self):

        self.scene.reset()
        observation = self.agent.reset()

        return observation

    def step(self, action):
        if self.scene_editor:
            self.user_interface.update_UI()
            self.scene.update_from_editor(self.user_interface)

        observation, reward, done, info = self.agent.step(action)
        
        if done:
            self.agent.reset()
            self.scene.reset()

        if self.debug:
            btn_val = p.readUserDebugParameter(self.reset_btn)
            # logging.debug(btn_val)
            if self.prev_reset_btn < btn_val:
                # logging.debug(
                #     f"Resetting scene  ({self.prev_reset_btn} < {btn_val})")
                self.prev_reset_btn = btn_val
                # self.scene.reset()
                self.reset()

        return observation, reward, done, info

    def run(self, seconds=1):
        # it should be run after reset and after setting joints
        logging.info('runnning simulation')
        for _ in range(int(seconds/(1/240))):
            # 1/240 seconds
            p.stepSimulation(self.physicsClient)