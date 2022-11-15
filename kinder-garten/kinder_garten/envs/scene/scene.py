import pybullet as p
import logging
import os
import random

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
        if 'dir' in object:
            spawner = SpawSquare(object)
            spawner.spawn()
            self.spawners.append(spawner)
        


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


    
        
            
