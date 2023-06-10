from enum import Enum
from kinder_garten.envs.camera.camera import PyBulletCamera
import gym
import numpy as np

from pybullet_utils import transformations

from dataclasses import dataclass

import logging

from ..bullet import Bullet

from kinder_garten.envs.agents.rewards import Reward


@dataclass
class ActionStateSpaces:
    maxTranslation: float = 0.01
    maxYawRotation: float = 0.1
    maxForce: float = 100


class Gripper:

    # config max_translation
    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        FAIL = 2
        TIME_LIMIT = 3

    def __init__(self, engine, physicsClient, reward, simplified=False, discrete=False, debug=False) -> None:
        self.grippersPartsId = (4, 5)

        self._max_translation = 0.01
        self._max_yaw_rotation = 0.1 # o 0.15

        self._discrete = discrete

        # TODO address this
        self._include_robot_height  = False

        self.debug = debug

        self.width = 64
        self.height = 64
        if engine == 'pybullet':
            # self.camera = camera.PyBulletCamera()
            self.physicsClient = physicsClient
            self.ops = Bullet(self.physicsClient, 'agents/gripper/gripper.sdf')
            self.camera = PyBulletCamera(self, self.width, self.height, K=None)

            if self.debug:
                self.ops.init_debug()      

        self._reward_fn = Reward(None, self)

        self._simplified = simplified

        self._gripper_open = True
        self.endEffectorAngle = 0.

        self._initial_height = self.ops.get_link(2)[0][2]

        # TODO documentar
        self.main_joints =  [0, 1, 2, 3]

        self._target_joint_pos = None

        self.episode_step = 0
        self.time_horizon = 150
        self.episode_rewards = np.zeros(self.time_horizon)

        self.setup_spaces()

    def get_pose(self):
        return self.ops.get_pose()

    def get_link(self, id):
        return self.ops.get_link(id)

    def get_joint(self, id):
        return self.ops.get_joint(id)

    def reset(self):
        self._gripper_open = True
        self._target_joint_pos = None
        self.endEffectorAngle = 0.
        self.episode_step = 0
        self.time_horizon = 150
        self.episode_rewards = np.zeros(self.time_horizon)

        observation = self.observe()

        self.ops.reset()
        return observation

    def set_position(self, joint_id, position, max_force=100.):
        self.ops.set_position(joint_id, position, max_force)

    def step_simulator(self, duration):
        # duration /= 10
        self.ops.step_simulator(duration)
        
    def setup_spaces(self):
        self.set_action_space()

        shape = self.camera.height, self.camera.width
        self.full_obs = False
        if self.full_obs:  # RGB + Depth + Actuator
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(shape[0], shape[1], 5))
        else:  # Depth
            # TODO cambiar imagen shape
            # Although SB3 supports both channel-last and channel-first images as input, we recommend using the channel-first convention when possible
            # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
            # self.observation_space = gym.spaces.Box(low=0, high=255,
            #                                         shape=(3, shape[0], shape[1]),
            #                                         dtype=np.uint8)

            self.observation_space = gym.spaces.Box(low=0, high=255,
                                        shape=(1,shape[0], shape[1]),
                dtype=np.uint8)
            
            # print(self.observation_space)


    def set_action_space(self):
        if self._discrete:
            self.action_space = gym.spaces.Discrete(11)

        else:
            self.action_space = gym.spaces.Box(-1.,
                                               1., shape=(5,), dtype=np.float32)
            logging.info("gym.spaces.Box(-1., 1., shape=(5,), dtype=np.float32)")
        
        self._act = self._full_act

        if self._include_robot_height:
            self._obs_scaler = np.array([1. / 0.05, 1.])
        else:
            self._obs_scaler = 1. / 0.1

    # TODO check if its really necesary
    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw = self._max_yaw_rotation
        elif yaw < -self._max_yaw_rotation:
            yaw = -self._max_yaw_rotation
        else:
            yaw = yaw

        return translation, yaw

    def _full_act(self, action):
        # logging.info(action)
        if not self._discrete:
            # Parse the action vector
            translation, yaw_rotation = self._clip_translation_vector(
                action[:3], action[3])
            # Open/close the gripper
            open_close = action[4]
        else:
            assert(isinstance(action, (np.int64, int)))
            x = [0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0, 0, 0][action]
            y = [0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0][action]
            z = [0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0][action]
            a = [0, 0, 0, 0, 0, 0, 0, self._yaw_step, -self._yaw_step, 0, 0][action]
            # Open/close the gripper
            open_close = [0, 0, 0, 0, 0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step][action]
            translation = [x, y, z]
            yaw_rotation = a

        # logging.debug(
        #     f'open_close: {open_close}, self._gripper_open {self._gripper_open}')
        # open
        if open_close > 0.: # and not self._gripper_open:
            self.open_close_gripper(close=False)
        # close
        elif open_close < 0.: # and self._gripper_open:
            self.open_close_gripper(close=True)
        # Move the robot

        self.relative_pose(translation, yaw_rotation)

    def open_close_gripper(self, close):
        # print(f'Closing: {close}')
        self._target_joint_pos = 0.3 if close else  0.5
        # print(f"self._target_joint_pos {self._target_joint_pos}")
        # left
        self.set_position(4, self._target_joint_pos if close  else -self._target_joint_pos)
        # right
        self.set_position(5, -self._target_joint_pos if close else self._target_joint_pos)


        self._gripper_open = not close
        self.step_simulator(1)

    def absolute_pose(self, target_pos, target_orn):
        # target_pos = self._enforce_constraints(target_pos)

        target_pos[2] = (target_pos[2] - self._initial_height)

        yaw = target_orn
        comp_pos = np.r_[target_pos, yaw]

        for i, joint in enumerate(self.main_joints):
            self.set_position(joint, comp_pos[i])

        self.step_simulator(0.1)

    def relative_pose(self, translation, yaw_rotation):
        # necesary because with the hand in an angle the sum to the axis are not straight
        pos, orn = self.ops.get_pose()
        _, _, yaw = transformations.euler_from_quaternion(orn)
        #Calculate transformation matrices
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = self.to_pose(T_world_new)

        self.absolute_pose(target_pos, self.endEffectorAngle)

    def to_pose(self, transform):
        translation = transform[:3, 3]
        rotation = transformations.quaternion_from_matrix(transform)
        return translation, rotation

    def step(self, action):
        self._act(action)

        new_obs = self.observe()
        reward, self.status = self._reward_fn()
        self.episode_rewards[self.episode_step] = reward

        if self.status != self.Status.RUNNING:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            # logging.info("time limit!")
            done, self.status = True, self.Status.TIME_LIMIT
        else:
            done = False

        self.episode_step += 1
        self.obs = new_obs
        # print("obs:")
        # print(self.obs)
        return self.obs, reward, done, dict()

    def get_gripper_width(self):
        """Query the current opening width of the gripper."""
        
        fr = self.get_joint(5)
        fl = self.get_joint(4)
        left_finger_pos = abs(0.3 - fl)
        right_finger_pos = abs(-0.3 - fr)        

        return left_finger_pos + right_finger_pos

    def observe(self):
        camera = True
        gripper_width = self.get_gripper_width()
        if camera:
            self.camera.set_relative_position()
            _, depth, _ = self.camera.get_img()

            sensor_pad  = []
            
            # obs_stacked = np.dstack((depth, sensor_pad))
            # return obs_stacked
            # depth = np.expand_dims(depth, axis=0).astype(np.uint8)
            return depth

    def object_detected(self, tol=0.1):
        """Grasp detection by checking whether the fingers stalled while closing."""
        
        gripper_width = self.get_gripper_width()
        
        # print(f"Tolerancia: {self._target_joint_pos == 0.3 and gripper_width }")
        # print(f'Gripper width: {gripper_width} {self._target_joint_pos}')
        return self._target_joint_pos == 0.3 and gripper_width > tol
