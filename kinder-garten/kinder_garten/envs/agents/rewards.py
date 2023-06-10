# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import numpy as np
import time


class Reward:
    """Simple reward function reinforcing upwards movement of grasped objects."""

    def __init__(self, config, robot):
        self._robot = robot
        # self._shaped = config.get('shaped', True)

        # self._max_delta_z = robot._actuator._max_translation
        # self._terminal_reward = config['terminal_reward']
        # self._grasp_reward = config['grasp_reward']
        # self._delta_z_scale = config['delta_z_scale']
        # self._lift_success = config.get('lift_success', self._terminal_reward)
        # self._time_penalty = config.get('time_penalty', False)
        # self._table_clearing = config.get('table_clearing', False)
        # self.lift_dist = None

        self._shaped = True

        # self._max_delta_z = robot._actuator._max_translation
        self._terminal_reward = 10000
        self._grasp_reward = 10
        #  how important is the delta z (up)
        self._delta_z_scale = 15000
        self._lift_success = self._terminal_reward # config.get('lift_success', self._terminal_reward)
        self._time_penalty = 100  # 100
        self._table_clearing = False
        self.lift_dist = 0.1

        # Placeholders
        self._lifting = False
        self._start_height = None
        self._old_robot_height = None

        self.reset()

        # plt.ion()  # Note this correction


        # fig = plt.figure()
        # plt.axis([0, 500, -1, 10])

        self.i = 0
        self.x = list()
        self.y = list()

    def __call__(self):
        position, _ = self._robot.get_pose()
        # print(f'Position: {position}, {position[2]}')
        robot_height = position[2]
        reward = 0.

        # si no toca la mesa y coge objeto
        if not position[2] < 0.76:
            # print("---------------------------------------------------------")

            if self._robot.object_detected(tol=0.12):
                # print("Object detected")
                if not self._lifting:
                    self._start_height = robot_height
                    self._lifting = True
                # print("\t", robot_height, self._start_height, self.lift_dist)
                if robot_height - self._start_height > self.lift_dist:
                    # Object was lifted by the desired amount
                    # print("\n DONE \n")
                    return self._terminal_reward, self._robot.Status.SUCCESS
                if self._shaped:
                    
                    # Intermediate rewards for grasping and lifting
                    delta_z = robot_height - self._old_robot_height
                    
                    lifting_reward = self._grasp_reward + self._delta_z_scale * delta_z
                    # print(f"doing shaped {delta_z} {lifting_reward}")
                    reward = lifting_reward
                    # if reward>300:
                    #     time.sleep(1)
                    #     print("High reward")

        else:
            self._lifting = False

        # Time penalty
        # if self._shaped:
        #     reward -= self._grasp_reward + self._delta_z_scale * self._max_delta_z
        # else:
        #     reward -= 0.01

        reward -= 5


        
        # self.x.append(self.i)
        # self.y.append(reward)
        # plt.scatter(self.i, reward)
        self.i += 1
        # plt.show()
        # if self.i % 50 == 0:
        #     plt.pause(0.000000000001)

        self._old_robot_height = robot_height
        # print(reward)
        return reward, self._robot.Status.RUNNING

    def reset(self):
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]
        self._lifting = False


