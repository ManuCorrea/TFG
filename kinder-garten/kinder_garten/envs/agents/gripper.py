from kinder_garten.envs.camera.camera import PyBulletCamera
import pybullet as p
import gym
import numpy as np

from pybullet_utils import transformations

from sklearn.preprocessing import MinMaxScaler

from dataclasses import dataclass

import logging

from pathlib import Path

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
        print(path)
        path2 = 'kinder_garten/envs/agents/gripper/gripper_a/gripper.sdf'
        self.load_agent_pybullet(path)

        if 1:
            p.setRealTimeSimulation(1)
        

    def get_pose(self):
        pos, orn, _, _, _, _ = p.getLinkState(self.agent_id, 3)
        return (pos, orn)

    def load_agent_pybullet(self, path, start_pos=[0, 0, 1],
                            start_orn=[0, 0, 0, 1.], scaling=1., static=False):
        self.start_pos = start_pos
        self.start_orn = start_orn
        logging.info(f'loading {path}')
        if path.endswith('.sdf'):
            self.agent_id = p.loadSDF(
                path, globalScaling=scaling, physicsClientId=self.physicsClient)[0]
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
        # for i in range(self.joints):
        #     p.setJointMotorControlArray(self.agent_id,
        #                                 jointIndices=self.joint_indices,
        #                                 controlMode=p.POSITION_CONTROL,
        #                                 targetPositions=[0] * self.joints,
        #                                 forces=self.forces)

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

    def __init__(self, engine, physicsClient, reward, simplified=False, discrete=False, debug=True) -> None:
        self.grippersPartsId = (4, 5)

        self._max_translation = 0.01
        self._max_yaw_rotation = 0.1 # o 0.15

        self._discrete = discrete

        # TODO address this
        self._include_robot_height  = False

        self.debug = debug

        if engine == 'pybullet':
            # self.camera = camera.PyBulletCamera()
            self.physicsClient = physicsClient
            self.ops = Bullet(self.physicsClient)
            self.camera = PyBulletCamera(self, 224, 171, K=None)

            if self.debug:
                self.ops.init_debug()      

        self._simplified = simplified

        self.action_space()

    def get_pose(self):
        return self.ops.get_pose()

    def reset(self):
        self.ops.reset()

    def normal_space(self):
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
        self._act = self._full_act

        if self._include_robot_height:
            self._obs_scaler = np.array([1. / 0.05, 1.])
            self.state_space = gym.spaces.Box(
                0., 1., shape=(2,), dtype=np.float32)
        else:
            self._obs_scaler = 1. / 0.1
            self.state_space = gym.spaces.Box(
                0., 1., shape=(1,), dtype=np.float32)

    def action_space(self):
        # Discrete [0, n+1]
        # Box n dimensional spaces of continuous space
        if self._simplified:
            self.simplified_action_space()
        else:
            self.normal_space()
        
        return self.action_space

    # TODO check if its really necesary
    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        yaw = self._max_yaw_rotation if yaw > self._max_yaw_rotation else yaw

        return translation, yaw

    def _full_act(self, action):
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
        if open_close > 0. and not self._gripper_open:
            self.robot.open_gripper()
            self._gripper_open = True
        elif open_close < 0. and self._gripper_open:
            self.robot.close_gripper()
            self._gripper_open = False
        # Move the robot
        else:
            return self.robot.relative_pose(translation, yaw_rotation)

    def absolute_pose(self, target_pos, target_orn):
        # target_pos = self._enforce_constraints(target_pos)

        # TODO YO weird
        # target_pos[0] *= -1
        target_pos[1] *= -1
        target_pos[2] = -1 * (target_pos[2] - self._initial_height)

        # _, _, yaw = transform_utils.euler_from_quaternion(target_orn)
        # yaw *= -1
        yaw = target_orn
        comp_pos = np.r_[target_pos, yaw]

        for i, joint in enumerate(self.main_joints):
            self._joints[joint].set_position(comp_pos[i])
            # does setJointMotorControl2

        self.run(0.1)

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

    def to_pose(transform):
        translation = transform[:3, 3]
        rotation = transformations.quaternion_from_matrix(transform)
        return translation, rotation

    def _simplified_act(self, action):
        if not self._discrete:
            # Parse the action vector
            translation, yaw_rotation = self._clip_translation_vector(
                action[:2], action[2])
        else:
            assert(isinstance(action, (np.int64, int)))
            if action < self.num_actions_pad:
                x = action / self.num_act_grains * self.trans_action_range - self._max_translation
                y = 0
                a = 0
            elif self.num_actions_pad <= action < 2*self.num_actions_pad:
                action -= self.num_actions_pad
                x = 0
                y = action / self.num_act_grains * self.trans_action_range - self._max_translation
                a = 0
            else:
                action -= 2*self.num_actions_pad
                x = 0
                y = 0
                a = action / self.num_act_grains * self.yaw_action_range - self._max_yaw_rotation
            translation = [x, y]
            yaw_rotation = a
        # Add constant Z offset
        translation = np.r_[translation, 0.005]
        # Move the robot
        return self.robot.relative_pose(translation, yaw_rotation)


    def step(self, action):
        if self.debug:
            self.ops.step_debug()
        else:
            self.action(action)
        self.observe()

    def observe(self):
        # pass robot position
        self.camera.set_relative_position()
        result = self.camera.get_img()
        return result

    def action(self, action):
        pass

    def step_simulator():
        p.step()

    def set_gripper_friction_pybullet(self, lateralFriction=1.15):
        for i in self.grippersPartsId:
            p.changeDynamics(self.agent_id, i, lateralFriction=lateralFriction)

    def get_joints(self):
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            self.joints[i] = self._physics_client.getJointInfo(self.model_id, i)


