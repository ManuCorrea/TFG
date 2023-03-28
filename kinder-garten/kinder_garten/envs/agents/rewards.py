# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import numpy as np

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
        self._terminal_reward = 10
        self._grasp_reward = 1
        #  how important is the delta z (up)
        self._delta_z_scale = 1000
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

        if self._robot.object_detected():
            print("Object detected")
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True
            print("\t", robot_height, self._start_height, self.lift_dist)
            if robot_height - self._start_height > self.lift_dist:
                print(
                    f'Got this up: {robot_height - self._start_height} must be higher thatn {self.lift_dist}')
                # Object was lifted by the desired amount
                print("\n\n\n\n\n\n DONE \n\n\n\n\n\n")
                return self._terminal_reward, self._robot.Status.SUCCESS
            if self._shaped:
                # Intermediate rewards for grasping and lifting
                delta_z = robot_height - self._old_robot_height
                reward = self._grasp_reward + self._delta_z_scale * delta_z

        else:
            self._lifting = False

        # Time penalty
        # if self._shaped:
        #     reward -= self._grasp_reward + self._delta_z_scale * self._max_delta_z
        # else:
        #     reward -= 0.01

        reward -= 0.1


        
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



class ShapedCustomReward(Reward):

    def __call__(self, obs, action, new_obs):
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.

        if self._robot.object_detected():
            # print("Object detected")
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True

            # print(
            #     f'Got this up: {robot_height - self._start_height} must be higher thatn {self.lift_dist}')
            if robot_height - self._start_height > self.lift_dist:
                if self._table_clearing:
                    # Object was lifted by the desired amount
                    grabbed_obj = self._robot.find_highest()
                    if grabbed_obj is not -1:
                        self._robot.remove_model(grabbed_obj)

                    # Multiple object grasping
                    # grabbed_objs = self._robot.find_higher(self.lift_dist)
                    # if grabbed_objs:
                    #     self._robot.remove_models(grabbed_objs)

                    self._robot.open_gripper()
                    if self._robot.get_num_body() == 2:
                        return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                    return self._lift_success, robot.RobotEnv.Status.RUNNING
                else:
                    if not self._shaped:
                        return 1., robot.RobotEnv.Status.SUCCESS
                    return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
            if self._shaped:
                # Intermediate rewards for grasping and lifting
                delta_z = robot_height - self._old_robot_height
                reward = self._grasp_reward + self._delta_z_scale * delta_z
        else:
            self._lifting = False

        # Time penalty
        if self._shaped:
            reward -= self._time_penalty
        else:
            reward -= 0.01

        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING
