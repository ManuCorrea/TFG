import pybullet as p
import logging
from pathlib import Path

class Bullet():
    def __init__(self, physicsClient, agent_path) -> None:
        self.physicsClient = physicsClient
        path = str(Path(__file__).parent/agent_path)
        logging.info(f'loading model from {path}')
        self.load_agent_pybullet(path, start_pos=[0,0,0.8])
        self._time_step = 1. / 240.

        if 1:
            self.physicsClient.setRealTimeSimulation(1)
        

    def get_pose(self):
        pos, orn, _, _, _, _ = self.physicsClient.getLinkState(self.agent_id, 3)
        return (pos, orn)

    def get_link(self, id):
        pos, orn, _, _, _, _ = self.physicsClient.getLinkState(self.agent_id, id)
        return (pos, orn)

    def get_joint(self, id):
        joint_state = self.physicsClient.getJointState(
            self.agent_id, id)
        return joint_state[0]

    def set_position(self, joint_id, position, max_force=100.):
        # print(f'Setting {joint_id} to {position}')
        self.physicsClient.setJointMotorControl2(
            self.agent_id, joint_id,
            controlMode=p.POSITION_CONTROL,
            # controlMode=p.VELOCITY_CONTROL,
            targetPosition=position,
            force=max_force,
            positionGain=0.02
            # targetVelocity=0.1,
            # velocityGain=0.1
            )

    def step_simulator(self, duration):
        for _ in range(int(duration / self._time_step)):
            self.physicsClient.stepSimulation()

    def load_agent_pybullet(self, path, start_pos=[0, 0, 1],
                            start_orn=[0, 0, 0, 1.], scaling=1., static=False):
        self.start_pos = start_pos
        self.start_orn = start_orn
        logging.info(f'loading {path}')
        if path.endswith('.sdf'):
            self.agent_id = self.physicsClient.loadSDF(
                path, globalScaling=scaling)[0]
            self.physicsClient.changeDynamics(self.agent_id, 4, lateralFriction=50)
            self.physicsClient.changeDynamics(self.agent_id, 5, lateralFriction=50)
            self.physicsClient.resetBasePositionAndOrientation(
                self.agent_id, start_pos, start_orn)
        else:
            self.agent_id = self.physicsClient.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static)

        self.joints = self.physicsClient.getNumJoints(self.agent_id)
        self.joint_indices = [idx for idx in range(self.joints)]
        self.forces = [60] * self.joints

    def reset(self):
        logging.debug(f"Resetting with pos: {self.start_pos} and orn {self.start_orn}")        
        
        self.physicsClient.setJointMotorControlArray(self.agent_id,
                                    jointIndices=self.joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * self.joints,
                                    forces=self.forces)



        self.physicsClient.resetBasePositionAndOrientation(
            self.agent_id, self.start_pos, self.start_orn)

        for idx in range(self.joints):
            # joint_states = self.physicsClient.getJointStates(self.agent_id,idx)
            self.physicsClient.resetJointState(self.agent_id, idx, 0)
            joint_state = self.physicsClient.getJointState(
                self.agent_id, idx)
        
        


    def init_debug(self):
        
        # print(self.physicsClient.getDynamicsInfo(self.agent_id, -1))

        self.sliders = []
        # self.joint_indices = []

        # for i in range(self.joints):
        #     print("-------------------------------")
        #     joint_info = self.physicsClient.getJointInfo(self.agent_id, i)
        #     joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
        #                     'force': joint_info[10]}
        #     index = joint_info[0]
        #     name = joint_info[1]
        #     print(index)
        #     print(name)
        #     print(joint_limits)

        #     slider = self.physicsClient.addUserDebugParameter(str(name), joint_limits["lower"],
        #                                     joint_limits["upper"], 0)
        #     self.sliders.append(slider)

        #     self.joint_indices.append(i)
        #     print("-------------------------------")

    def step_debug(self):
        
        joint_values = []
        for i in range(self.joints):
            joint_values.append(self.physicsClient.readUserDebugParameter(self.sliders[i]))
        self.physicsClient.setJointMotorControlArray(self.agent_id,
                                    jointIndices=self.joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_values,
                                    forces=self.forces)