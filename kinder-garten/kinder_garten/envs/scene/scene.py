import pybullet as p
import logging
import os
import random
import numpy  as np
from kinder_garten.envs.agents.gripper  import Bullet
import math
from ...GUI.utils import UserInterface

# logging.basicConfig(filename='gripper.log', level=logging.DEBUG,
#                     format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d:%H:%M:%S')


# # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# red = "\x1b[31;20m"
# reset = "\x1b[0m"

# logFormatter = logging.Formatter(
#     red + "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s"  + reset)

# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# logging.getLogger().addHandler(consoleHandler)

class Scene:
    def __init__(self, engine, scene, client) -> None:
        
        self.client = client
        if engine == 'pybullet':
            self.initPyBullet()
            self.load = self.loadPyBullet
            self.reset = self.resetPyBullet

        if scene == 'simple':
            from kinder_garten.envs.scene.simple import objects_to_load
            self.objects_to_load = objects_to_load
        elif scene == 'table':
            from kinder_garten.envs.scene.mesa import objects_to_load
            self.objects_to_load = objects_to_load
        elif scene == 'editor':
            from kinder_garten.envs.scene.simple import objects_to_load
            self.objects_to_load = objects_to_load
            self.added_spawn_areas = set()

        for object in objects_to_load:
            self.load(object)

            
    def initPyBullet(self):
        import pybullet_data

        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

    def get_pose(self, id):
        pos, orn = self.client.getBasePositionAndOrientation(id)
        return (pos, orn)

    def loadPyBullet(self, object):
        self.spawners = []
        if 'file' in object:
            file = object['file']
            logging.info(f'Loading {file}')

            if file.endswith('.sdf'):
                model_id = self.client.loadSDF(file)[0]
            else:
                id = self.client.loadURDF(file)
                # id = self.clientId.loadURDF(file)
            if 'detect_surface' in object and object['detect_surface']:
                self.detect_surface(id)
            
        if 'dir' in object:
            spawner = SpawnSquare(object, self.client)
            spawner.spawn()
            self.spawners.append(spawner)
        
    def detect_surface(self, obj_id):
        pos, _ = self.get_pose(obj_id)
        x = pos[0]
        y = pos[1]
        z = pos[2]

        rayFrom = (x, y, z+1)
        rayTo = (x, y, z-1)
        rayInfo = self.client.rayTest(rayFrom, rayTo)
        hit = rayInfo[0][3]
        z = hit[2]

        self.client.addUserDebugLine(rayFrom, rayTo, [1, 0, 0], 3)
        print(f'{rayInfo} len: {len(rayInfo)}')
        self.client.addUserDebugPoints([hit], [[0, 1, 1]], pointSize=10)

        xs = iter(np.linspace(-1, 1, 30))
        ys = iter(np.linspace(-1, 1, 30))


        rays_from = []
        rays_to = []
        table_planes = []
        logging.info('generating points')
        # TODO do surface generation space accorging object
        for x in np.linspace(-1, 1, 100):
            for y in np.linspace(-1, 1, 100):
                rayFrom = (x, y, z+0.1)
                rayTo = (x, y, z-0.1)
                rays_from.append(rayFrom)
                rays_to.append(rayTo)
                # rayInfo = self.client.rayTest(rayFrom, rayTo)
                # self.client.addUserDebugLine(rayFrom,rayTo,[1,0,0],3)
                # print(f'{rayInfo} len: {len(rayInfo)}')
                
                # hit = rayInfo[0]
                # objectUid = hit[0]
                # if (objectUid == obj_id):
                #     self.client.addUserDebugPoints(
                #         [rayFrom], [[0, 0, 1]], pointSize=5)

                #     pos_table = hit[3]
                #     table_planes.append(pos_table)
                # else:
                #     self.client.addUserDebugPoints(
                #         [rayFrom], [[0, 1, 0]], pointSize=5)

        logging.info('ray request')

        batch = self.client.rayTestBatch(rays_from, rays_to)
        # print(batch)

        logging.info('analyzing data points')
        for data in batch:
            objectUid = data[0]
            if (objectUid == obj_id):

                pos_table = data[3]

                table_planes.append(pos_table)

        # print(len(table_planes))
        # print(len([[0, 0, 1]]*len(table_planes)))
        # print([[0, 0, 1]]*len(table_planes))

        logging.info('painting points')
        # self.client.addUserDebugPoints(
        #     table_planes, [[0, 0, 1]]*len(table_planes), pointSize=5)

        x_min= 10000
        y_min = 10000
        x_max= -10000
        y_max = -10000
        for point in table_planes:
            x, y, _ = point
            x_min = x if x < x_min else x_min
            y_min = y if y < y_min else y_min
            x_max = x if x > x_max else x_max
            y_max = y if y > y_max else y_max

        # self.client.addUserDebugPoints(
        #     [[x_min, y_min, z]], [[0, 0, 0]], pointSize=10)
        # self.client.addUserDebugPoints(
        #     [[x_min, y_max, z]], [[0, 0, 0]], pointSize=10)
        # self.client.addUserDebugPoints(
        #     [[x_max, y_min, z]], [[0, 0, 0]], pointSize=10)
        # self.client.addUserDebugPoints(
        #     [[x_max, y_max, z]], [[0, 0, 0]], pointSize=10)
            

        
    def form_rectangle(self, pts):
        # find one of the corners
        for i in range(len(pts)):
            pt1 = pts[1]
            for j in range(i, len(pts)):
                pt2 = pts[j]
                x1, y1 = pt1
                x2, y2 = pt2
                slope = (y2-y1)/(x2-x1)
                print(f'{pt1} {pt2} m:{slope}')


    def resetPyBullet(self):
        for object in self.objects_to_load:
            if object['reset']:
                # move to original position
                # this shouldn't handle agent, just objects
                pass

        for spawner in self.spawners:
            spawner.reset()

    # TODO how design approach
    def isDone(self):
        return False

    def update_from_editor(self, user_interface: UserInterface):
        # TODO decouple logic from canvas
        # vamos a coger los datos y dibujarlos
        # create aux list for proceced lines so we dont over draw

        # Lines which represent walls
        for line in user_interface.get_canvas_lines():
            p1 = list(line[0])
            p2 = list(line[1])
            # add z dimension
            z = 200

            p1.append(z)
            p2.append(z)

            scale = 0.01

            p1 = np.array(p1) * scale
            p2 = np.array(p2) * scale

            mid_point = (p1+p2)/2

            vector = p1-p2

            angle = math.degrees(math.atan(vector[1]/vector[0]))

            if 90.0 == abs(angle):
                orientation = self.client.getQuaternionFromEuler([1.57, 0, 1.57])
            else:
                orientation = self.client.getQuaternionFromEuler([1.57, 0, 0])

            print(f'----- {angle}')
            magnitude = np.sqrt(vector.dot(vector))
            # magnitude *= 0.5

            mid_point[2] = 0.5
            
            # middle of the plane in between the 2 planes
            # plane1x1 = "kinder-garten/kinder_garten/envs/scene/objects/plane/plane_1x1.urdf"
            # id = self.client.loadURDF(plane1x1, basePosition=mid_point, baseOrientation=orientation, globalScaling=magnitude)
            meshScale = [1*magnitude, 1, 3]
            shift = [0, -0.02, 0]
            visualShapeId = self.client.createVisualShape(shapeType=self.client.GEOM_MESH,
                                                          fileName="/home/yo/Desktop/Desarrollo/TFG/kinder-garten/kinder_garten/envs/scene/objects/plane/a.obj",
                                    rgbaColor=[1, 0, 0, 1],
                                    specularColor=[0.4, .4, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)

            # TODO add collision shape

            self.client.createMultiBody(baseMass=0,
                      baseInertialFramePosition=[0, 0, 0],
                    #   baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      baseOrientation=orientation,
                      basePosition=mid_point,
                      useMaximalCoordinates=True)

            # print(f'drawing lines {p1} {p2}')
            self.client.addUserDebugLine(p1, p2, lineColorRGB=[1,1,0], lineWidth=5)

        for spawn_area in user_interface.get_spawn_areas():
            id, x1, y1, x2, y2, group = spawn_area
            if id not in self.added_spawn_areas:
                self.added_spawn_areas.add(id)
                SpawnRectangle(spawn_area, self.client)
            

            # visualShapeId = self.client.createVisualShape(shapeType=self.client.GEOM_MESH,
            #                                     fileName="/home/yo/Desktop/Desarrollo/TFG/kinder-garten/kinder_garten/envs/scene/objects/plane/a.obj",
            #             rgbaColor=[1, 0, 0, 1],
            #             specularColor=[0.4, .4, 0],
            #             visualFramePosition=shift,
            #             meshScale=meshScale)
            
            # self.client.createMultiBody(baseMass=0,
            #           baseInertialFramePosition=[0, 0, 0],
            #         #   baseCollisionShapeIndex=collisionShapeId,
            #           baseVisualShapeIndex=visualShapeId,
            #           baseOrientation=orientation,
            #           basePosition=mid_point,
            #           useMaximalCoordinates=True)
            
            """
            self.dir = obj['dir']
            position = obj['position']
            self.position = position
            size = obj['size']
            self._reset = obj['reset']
            self.client = client

            size /= 2
            self.size = size
            """


        # TODO add spawning area processing

class SpawnSquare:
    def __init__(self, obj, client, debug=True) -> None:
        self.dir = obj['dir']
        position = obj['position']
        self.position = position
        size = obj['size']
        self._reset = obj['reset']
        self.client = client

        size /= 2
        self.size = size

        if debug:
            p1 = (position[0] - size, position[1] - size, position[2])
            p2 = (position[0] - size, position[1] + size, position[2])
            p3 = (position[0] + size, position[1] - size, position[2])
            p4 = (position[0] + size, position[1] + size, position[2])
            self.client.addUserDebugLine(p1, p2)
            self.client.addUserDebugLine(p1, p3)
            self.client.addUserDebugLine(p2, p4)
            self.client.addUserDebugLine(p3, p4)

    def spawn(self):
        self.loaded_objs = set()
        if os.path.isdir(self.dir):
            for obj in os.listdir(self.dir):
                if obj.endswith('.urdf'):
                    pos = (self.position[0] + self.size * random.uniform(-1, 1),
                           self.position[1] + self.size * random.uniform(-1, 1),
                           self.position[2])
                    id = self.client.loadURDF(
                        f'{self.dir}/{obj}', pos, useFixedBase=False)
                    # print(f'Object with id {id}')
                    self.loaded_objs.add(id)

    def reset(self):
        if self._reset:
            for obj in self.loaded_objs:
                # print(f'Remove obj with id {obj}')
                self.client.removeBody(obj)
            self.loaded_objs = set()
        self.spawn()


class SpawnRectangle:
    def __init__(self, spawn_area, client, debug=True) -> None:
        _, x1, y1, x2, y2, group = spawn_area
        _, self.x1, self.y1, self.x2, self.y2, self.group = spawn_area

        scale = 0.01
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        self.x1, self.y1, self.x2, self.y2 = self.x1 * scale, self.y1 * scale, self.x2 * scale, self.y2 * scale

        self.dir = 0
        position = 0
        self.position = position
        self.client = client

        z = 1
        self.z = z

        self.x_lenght = abs(x1-x2)
        self.y_lenght = abs(y1-y2)

        if debug:
            p1 = (x1, y1, z)
            p2 = (x1+self.x_lenght, y1, z)
            p3 = (x1, y1+self.y_lenght, z)
            p4 = (x2, y2, z)
            self.client.addUserDebugLine(p1, p2)
            self.client.addUserDebugLine(p1, p3)
            self.client.addUserDebugLine(p2, p4)
            self.client.addUserDebugLine(p3, p4)

    def spawn(self):
        self.loaded_objs = set()
        # if os.path.isdir(self.dir):
        #     for obj in os.listdir(self.dir):
        #         if obj.endswith('.urdf'):
        pos = (self.x1 + self.x_lenght * random.uniform(0, 1),
                self.y1 + self.y_lenght * random.uniform(0, 1),
                self.z)
        id = self.client.loadURDF(
            "r2d2.urdf", pos, useFixedBase=False)
        # print(f'Object with id {id}')
        self.loaded_objs.add(id)

    def reset(self):
        if self._reset:
            for obj in self.loaded_objs:
                # print(f'Remove obj with id {obj}')
                self.client.removeBody(obj)
            self.loaded_objs = set()
        self.spawn()

