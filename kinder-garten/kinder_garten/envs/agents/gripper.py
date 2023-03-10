from enum import Enum
from kinder_garten.envs.camera.camera import PyBulletCamera
import pybullet as p
import gym
import numpy as np

from pybullet_utils import transformations

from sklearn.preprocessing import MinMaxScaler

from dataclasses import dataclass

import logging

from pathlib import Path


from kinder_garten.envs.agents.rewards import Reward, SimplifiedReward, ShapedCustomReward


# logging.basicConfig(filename='gripper.log', level=logging.DEBUG,
#                     format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d:%H:%M:%S')

"""
continuous or discrete
"""

@dataclass
class ActionStateSpaces:
    maxTranslation: float = 0.01
    maxYawRotation: float = 0.1
    maxForce: float = 100

class Bullet():
    def __init__(self, physicsClient) -> None:
        self.physicsClient = physicsClient
        path = str(Path(__file__).parent/'gripper/gripper.sdf')
        logging.info(f'loading model from {path}')
        self.load_agent_pybullet(path, start_pos=[0,0,0.8])
        self._time_step = 1. / 240.

        if 1:
            p.setRealTimeSimulation(1)
        

    def get_pose(self):
        pos, orn, _, _, _, _ = p.getLinkState(self.agent_id, 3)
        return (pos, orn)

    def get_link(self, id):
        pos, orn, _, _, _, _ = p.getLinkState(self.agent_id, id)
        return (pos, orn)

    def get_joint(self, id):
        joint_state = p.getJointState(
            self.agent_id, id)
        return joint_state[0]

    def set_position(self, joint_id, position, max_force=100.):
        # print(f'Setting {joint_id} to {position}')
        p.setJointMotorControl2(
            self.agent_id, joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            force=max_force)

    def step_simulator(self, duration):
        for _ in range(int(duration / self._time_step)):
            p.stepSimulation()

    def load_agent_pybullet(self, path, start_pos=[0, 0, 1],
                            start_orn=[0, 0, 0, 1.], scaling=1., static=False):
        self.start_pos = start_pos
        self.start_orn = start_orn
        logging.info(f'loading {path}')
        if path.endswith('.sdf'):
            self.agent_id = p.loadSDF(
                path, globalScaling=scaling, physicsClientId=self.physicsClient)[0]
            p.changeDynamics(self.agent_id, 4, lateralFriction=10)
            p.changeDynamics(self.agent_id, 5, lateralFriction=10)
            p.resetBasePositionAndOrientation(
                self.agent_id, start_pos, start_orn, physicsClientId=self.physicsClient)
        else:
            self.agent_id = p.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static, physicsClientId=self.physicsClient)

        self.joints = p.getNumJoints(self.agent_id)
        self.forces = [100] * self.joints

    def reset(self):
        logging.debug(f"Resetting with pos: {self.start_pos} and orn {self.start_orn}")
        # p.resetBasePositionAndOrientation(
        #     self.agent_id, self.start_pos, self.start_orn, physicsClientId=self.physicsClient)
        for i in range(self.joints):
            p.setJointMotorControlArray(self.agent_id,
                                        jointIndices=self.joint_indices,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0] * self.joints,
                                        forces=self.forces)


        p.resetBasePositionAndOrientation(
            self.agent_id, self.start_pos, self.start_orn, physicsClientId=self.physicsClient)
        for idx in self.joint_indices:
            p.resetJointState(self.agent_id, idx, 0)
        


    def init_debug(self):
        
        print(p.getDynamicsInfo(self.agent_id, -1))

        self.sliders = []
        self.joint_indices = []

        for i in range(self.joints):
            print("-------------------------------")
            joint_info = p.getJointInfo(self.agent_id, i)
            joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
                            'force': joint_info[10]}
            index = joint_info[0]
            name = joint_info[1]
            print(index)
            print(name)
            print(joint_limits)

            slider = p.addUserDebugParameter(str(name), joint_limits["lower"],
                                            joint_limits["upper"], 0, physicsClientId=self.physicsClient)
            self.sliders.append(slider)

            self.joint_indices.append(i)
            print("-------------------------------")

    def step_debug(self):
        
        joint_values = []
        for i in range(self.joints):
            joint_values.append(p.readUserDebugParameter(self.sliders[i]))
        p.setJointMotorControlArray(self.agent_id,
                                    jointIndices=self.joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_values,
                                    forces=self.forces)


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

        self.width = 224
        self.height = 171
        if engine == 'pybullet':
            # self.camera = camera.PyBulletCamera()
            self.physicsClient = physicsClient
            self.ops = Bullet(self.physicsClient)
            self.camera = PyBulletCamera(self, self.width, self.height, K=None)

            if self.debug:
                self.ops.init_debug()      

        self._reward_fn = Reward(None, self)

        self._simplified = simplified

        self._gripper_open = True
        self.endEffectorAngle = 0.

        print(f'---------------  eje  z: {self.ops.get_link(2)[0][2]}')
        print(f'---------------  base: {self.ops.get_link(3)[0]}')
        # TODO double check
        self._initial_height = self.ops.get_link(2)[0][2]

        # TODO documentar
        self.main_joints =  [0, 1, 2, 3]

        # TODO get sure about this value
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

        self.ops.reset()

    def set_position(self, joint_id, position, max_force=100.):
        self.ops.set_position(joint_id, position, max_force)

    def step_simulator(self, duration):
        # duration /= 10
        self.ops.step_simulator(duration)
        
    def setup_spaces(self):
        self.set_action_space()

        shape = self.camera.width, self.camera.height
        self.full_obs = False
        if self.full_obs:  # RGB + Depth + Actuator
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(shape[0], shape[1], 5))
        else:  # Depth + Actuator obs
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(shape[0], shape[1], 2))


    def set_action_space(self):
        if self._discrete:

            # [no, -x, +x, -y, +y, -z, +z, -yaw, +yaw, open, close]
            self.action_space = gym.spaces.Discrete(11)
            #TODO Implement the linear discretization for full environment
            # self.action_space = gym.spaces.Discrete(self.num_actions_pad*5) # +1 to add no action(zero action)
        else:
            high = np.array([self._max_translation,
                self._max_translation,
                self._max_translation,
                self._max_yaw_rotation,
                1.])

            self._action_scaler = MinMaxScaler((-1, 1))
            self._action_scaler.fit(np.vstack((-1. * high, high)))

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
        else:
            self.relative_pose(translation, yaw_rotation)

    def open_close_gripper(self, close):
        print(f'Closing: {close}')
        self._target_joint_pos = 0.3 if close else  0.6
        # left
        self.set_position(4, self._target_joint_pos if close  else -self._target_joint_pos)
        # right
        self.set_position(5, -self._target_joint_pos if close else self._target_joint_pos)


        self._gripper_open = not close
        self.step_simulator(0.2)

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
        # print(f'---------------  eje  z: {self.ops.get_link(2)[0]}')
        # print(f'---------------  base: {self.ops.get_link(3)[0]}')
        
        # if self.debug:
        #     self.ops.step_debug()
        # else:
        #     self.action(action)
        
        # if self._model is None:
        #     self.reset()

        # TODO implementar
        self.action(action)

        new_obs = self.observe()
        reward, self.status = self._reward_fn()
        self.episode_rewards[self.episode_step] = reward

        if self.status != self.Status.RUNNING:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            logging.info("time limit!")
            done, self.status = True, self.Status.TIME_LIMIT
        else:
            done = False

        # if done:
        #     self._trigger_event(self.Status.END_OF_EPISODE, self)

        self.episode_step += 1
        self.obs = new_obs

        return self.obs, reward, done

    def get_gripper_width(self):
        """Query the current opening width of the gripper."""
        
        fr = self.get_joint(5)
        fl = self.get_joint(4)
        left_finger_pos = abs(0.3 - fl)
        right_finger_pos = abs(-0.3 - fr)        

        # logging.debug(
        #     f'Position R finger: {fr}, Position L finger: {fl} target positions {right_finger_pos, left_finger_pos,} \n Total width: {left_finger_pos + right_finger_pos}')

        return left_finger_pos + right_finger_pos

    def observe(self):
        camera = True
        gripper_width = self.get_gripper_width()
        if camera:
            self.camera.set_relative_position()
            rgb, depth, _ = self.camera.get_img()

            sensor_pad  = []
            
            # obs_stacked = np.dstack((depth, sensor_pad))
            # return obs_stacked
            return depth

    def object_detected(self, tol=0.1):
        """Grasp detection by checking whether the fingers stalled while closing."""
        # target joint position is on closing gripper
        # and
        # lo mas cercano a cero es que esta cerrado
        # tol is the minimum with of an object
        gripper_width = self.get_gripper_width()
        # print(f'Gripper width: {gripper_width}')
        # no funciona porque no se mantiene cerrada
        return self._target_joint_pos == 0.3 and gripper_width > tol

    def action(self, action):
        self._act(action)
