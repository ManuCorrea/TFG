import pybullet as p
import logging
import os
import random
import numpy  as np
from kinder_garten.envs.agents.gripper  import Bullet

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
    def __init__(self, engine, scene, clientId) -> None:
        
        self.clientId = clientId
        if engine is 'pybullet':
            self.initPyBullet()
            self.load = self.loadPyBullet
            self.reset = self.resetPyBullet

        if scene is 'simple':
            from kinder_garten.envs.scene.simple import objects_to_load
            self.objects_to_load = objects_to_load
        elif scene is 'table':
            from kinder_garten.envs.scene.mesa import objects_to_load
            self.objects_to_load = objects_to_load

        for object in objects_to_load:
            self.load(object)

            
    def initPyBullet(self):
        import pybullet_data

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def get_pose(self, id):
        pos, orn = p.getBasePositionAndOrientation(id)
        return (pos, orn)

    def loadPyBullet(self, object):
        self.spawners = []
        if 'file' in object:
            file = object['file']
            logging.info(f'Loading {file}')

            if file.endswith('.sdf'):
                model_id = p.loadSDF(file)[0]
            else:
                # id = p.loadURDF(file, physicsClientId=self.clientId)
                id = p.loadURDF(file)
            if 'detect_surface' in object and object['detect_surface']:
                self.detect_surface(id)
            
        if 'dir' in object:
            spawner = SpawSquare(object)
            spawner.spawn()
            self.spawners.append(spawner)
        
    def detect_surface(self, obj_id):
        pos, _ = self.get_pose(obj_id)
        x = pos[0]
        y = pos[1]
        z = pos[2]

        rayFrom = (x, y, z+1)
        rayTo = (x, y, z-1)
        rayInfo = p.rayTest(rayFrom, rayTo)
        hit = rayInfo[0][3]
        z = hit[2]

        p.addUserDebugLine(rayFrom, rayTo, [1, 0, 0], 3)
        print(f'{rayInfo} len: {len(rayInfo)}')
        p.addUserDebugPoints([hit], [[0, 1, 1]], pointSize=10)

        xs = iter(np.linspace(-1, 1, 30))
        ys = iter(np.linspace(-1, 1, 30))


        rays_from = []
        rays_to = []
        table_planes = []
        logging.info('generating points')
        for x in np.linspace(-1, 1, 10000):
            for y in np.linspace(-1, 1, 10000):
                rayFrom = (x, y, z+0.1)
                rayTo = (x, y, z-0.1)
                rays_from.append(rayFrom)
                rays_to.append(rayTo)
                # rayInfo = p.rayTest(rayFrom, rayTo)
                # p.addUserDebugLine(rayFrom,rayTo,[1,0,0],3)
                # print(f'{rayInfo} len: {len(rayInfo)}')
                
                # hit = rayInfo[0]
                # objectUid = hit[0]
                # if (objectUid == obj_id):
                #     p.addUserDebugPoints(
                #         [rayFrom], [[0, 0, 1]], pointSize=5)

                #     pos_table = hit[3]
                #     table_planes.append(pos_table)
                # else:
                #     p.addUserDebugPoints(
                #         [rayFrom], [[0, 1, 0]], pointSize=5)

        logging.info('ray request')

        batch = p.rayTestBatch(rays_from, rays_to)
        print(batch)
        points = []

        logging.info('analyzing data points')
        for data in batch:
            objectUid = data[0]
            if (objectUid == obj_id):

                pos_table = data[3]

                table_planes.append(pos_table)

        print(len(table_planes))
        print(len([[0, 0, 1]]*len(table_planes)))
        print([[0, 0, 1]]*len(table_planes))

        logging.info('painting points')
        p.addUserDebugPoints(
            table_planes, [[0, 0, 1]]*len(table_planes), pointSize=5)

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

        p.addUserDebugPoints(
            [[x_min, y_min, z]], [[0, 0, 0]], pointSize=10)
        p.addUserDebugPoints(
            [[x_min, y_max, z]], [[0, 0, 0]], pointSize=10)
        p.addUserDebugPoints(
            [[x_max, y_min, z]], [[0, 0, 0]], pointSize=10)
        p.addUserDebugPoints(
            [[x_max, y_max, z]], [[0, 0, 0]], pointSize=10)
            

        




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


class SpawSquare:
    def __init__(self, obj, debug=True) -> None:
        self.dir = obj['dir']
        position = obj['position']
        self.position = position
        size = obj['size']
        self._reset = obj['reset']

        size /= 2
        self.size = size

        if debug:
            p1 = (position[0] - size, position[1] - size, position[2])
            p2 = (position[0] - size, position[1] + size, position[2])
            p3 = (position[0] + size, position[1] - size, position[2])
            p4 = (position[0] + size, position[1] + size, position[2])
            p.addUserDebugLine(p1, p2)
            p.addUserDebugLine(p1, p3)
            p.addUserDebugLine(p2, p4)
            p.addUserDebugLine(p3, p4)

    def spawn(self):
        self.loaded_objs = set()
        if os.path.isdir(self.dir):
            for obj in os.listdir(self.dir):
                if obj.endswith('.urdf'):
                    pos = (self.position[0] + self.size * random.uniform(-1, 1),
                           self.position[1] + self.size * random.uniform(-1, 1),
                           self.position[2])
                    id = p.loadURDF(
                        f'{self.dir}/{obj}', pos, useFixedBase=False)
                    print(f'Object with id {id}')
                    self.loaded_objs.add(id)

    def reset(self):
        if self._reset:
            for obj in self.loaded_objs:
                print(f'Remove obj with id {obj}')
                p.removeBody(obj)
            self.loaded_objs = set()
        self.spawn()


    
        
            
