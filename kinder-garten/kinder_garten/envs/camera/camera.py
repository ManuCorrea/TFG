from turtle import width
import pybullet as p
import math
import numpy as np
from dataclasses import dataclass

import logging

class OglOps():
    @staticmethod
    def _gl_ortho(left, right, bottom, top, near, far):
        """Implementation of OpenGL's glOrtho subroutine."""
        ortho = np.diag(
            [2./(right-left), 2./(top-bottom), - 2./(far-near), 1.])
        ortho[0, 3] = - (right + left) / (right - left)
        ortho[1, 3] = - (top + bottom) / (top - bottom)
        ortho[2, 3] = - (far + near) / (far - near)
        return ortho


    @staticmethod
    def _build_projection_matrix(height, width, K, near, far):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        perspective = np.array([[fx, 0., -cx, 0.],
                                [0., fy, -cy, 0.],
                                [0., 0., near + far, near * far],
                                [0., 0., -1., 0.]])
        ortho = OglOps._gl_ortho(0., width, height, 0., near, far)
        return np.matmul(ortho, perspective)

    def quaternion_matrix(quaternion):
        """Return homogeneous rotation matrix from quaternion.

        >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True

        """
        # epsilon for testing whether a number is close to zero
        _EPS = np.finfo(float).eps * 4.0

        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1] -
             q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    
    @staticmethod
    def from_pose(translation, rotation, invert=False):
        """Create a transform from a translation vector and quaternion.

        Args:
            translation: A translation vector.
            rotation: A quaternion in the form [x, y, z, w].

        Returns:
            A 4x4 homogeneous transformation matrix.
        """
        transform = OglOps.quaternion_matrix(rotation)
        transform[:3, 3] = translation
        return transform


@dataclass
class ProjectionMatrixProps:
    width: int
    height: int
    K: float
    near: float
    far: float

class PyBulletCamera():
    """
    Use:
    init()
    set
    """
    def __init__(self, agent, width, height, K, near=0.02, far=2.0, debug=False) -> None:
        self.width = width
        self.height = height
        self._near = near
        self._far = far
        self.view_matrix = None
        self.agent = agent
        self.debug = debug
        # K = np.array(
        #     [[69.76, 0.0, 32.19], [0.0, 77.25, 32.0], [0.0, 0.0, 1.0]])

        # https: // ksimek.github.io/2013/08/13/intrinsic/
        K = np.array(
            [[186.400, 0.0, 112.0], [0.0, 206.415, 85.5], [0.0, 0.0, 1.0]]
        )
        self.projection_matrix = OglOps._build_projection_matrix(
            height, width, K, near, far) #.flatten(order='F')
        print('\n\n\n')
        print(self.projection_matrix)
        mult = 1000
        otra = p.computeProjectionMatrix(
            0., width, height, 0., 0.02, 2.)

        # self.projectionMatrix = p.computeProjectionMatrixFOV(
        #     fov=45.0,
        #     aspect=width/height,
        #     nearVal=0.1,
        #     farVal=3.1)

        print(otra)
        print(np.array(otra))
        print('\n\n\n')


        self.init_camera_position()

        self.init_debug()

    def init_debug(self):
        self.args = ['t_x', 't_y', 't_z', 'rotation_x',
                     'rotation_y', 'rotation_z', 'rotation_w']
        
        self.sliders = {}
        for i in self.args:
            self.sliders[i] = p.addUserDebugParameter(i, -3.14, 3.14, 0)

        # self.sliders['near'] = p.addUserDebugParameter('near', 0.01, 10000, 0)
        # self.sliders['far'] = p.addUserDebugParameter('far', 0.1, 50000, 0)


    def get_debug_vals(self):
        data  = {}
        for i in self.args:
            data[i] = p.readUserDebugParameter(self.sliders[i])
        x = data['rotation_x']
        y = data['rotation_y']
        z = data['rotation_z']
        w = data['rotation_w']

        t_x = data['t_x']
        t_y = data['t_y']
        t_z = data['t_z']


        # near = p.readUserDebugParameter(self.sliders['near'])
        # far = p.readUserDebugParameter(self.sliders['far'])
        # self.projection_matrix = p.computeProjectionMatrix(
        #     0., self.width, self.height, 0., near, far)

        self.init_camera_position(translation=[t_x, 0.06, t_z], rotation=[0.6,y,z,0.06])


    def set_world_position(self):
        pass

    # Borrar. Creates _h_robot_camera
    def init_camera_position(self, translation = [0.0, 0.2, 0.05],  rotation = [1.0, -0.1305, 0.0, 0.0], transform=[]):
        "Generates the matrix trasformation for later use as relative to X point"

        # rotation = p.getQuaternionFromEuler([0, 1.57, 0])

        transform =  OglOps.from_pose(translation, rotation)

        self._h_robot_camera = transform

    def set_relative_position(self):
        """
        Sets the position of the camera from the robot pose
        """
        self.get_debug_vals()
        # we need here the position of our robot or X since the camera will be relative to it

        h_world_robot = OglOps.from_pose(
            *self.agent.get_pose())

        drawCam = True
        if drawCam:
            cam_matrix = np.dot(h_world_robot, self._h_robot_camera)
            cam_pos = cam_matrix[:3, 3]
        
            # logging.debug(f'pos {cam_pos} rot {cam_matrix[:3, 2]}')
            
            lineFrom = cam_pos.tolist()
            lineFrom = [round(point,3) for point in lineFrom]
            lineTo = lineFrom + lineFrom*cam_matrix[:3, 2]
            # logging.debug(f'vec from {lineFrom} to {lineTo}')
            if self.debug:
                p.addUserDebugPoints([cam_pos], [[0.1, 0, 0]],
                                    pointSize=3, lifeTime=0.1)
                p.addUserDebugLine(lineFrom, lineTo, lifeTime=0.1)


        # https://computergraphics.stackexchange.com/questions/5571/why-do-i-need-to-inverse-the-orientation-matrix-of-a-camera-to-be-able-to-transl
        # People always forget that there is no "camera" in OpenGL. In order to simulate a camera you have to move the whole world inversely
        h_camera_world = np.linalg.inv(
            np.dot(h_world_robot, self._h_robot_camera))

        self.view_matrix = h_camera_world

    def get_img(self):

        gl_view_matrix = self.view_matrix.copy()
        gl_view_matrix[2, :] *= -1  # flip the Z axis to comply to OpenGL
        gl_view_matrix = gl_view_matrix.flatten(order='F')

        gl_projection_matrix = self.projection_matrix.flatten(order='F')
        # gl_projection_matrix = self.projection_matrix

        result = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_projection_matrix,
            renderer=p.ER_TINY_RENDERER)

        # Extract RGB image
        rgb = np.asarray(result[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (self.height, self.width, 4))[:, :, :3]
        # Extract depth image
        near, far = self._near, self._far
        depth_buffer = np.asarray(result[3], np.float32).reshape(
            (self.height, self.width))
        depth = 1. * far * near / (far - (far - near) * depth_buffer)

        # Extract segmentation mask
        mask = result[4]

        return rgb, depth, mask



