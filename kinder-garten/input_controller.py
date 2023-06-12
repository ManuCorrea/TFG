from xbox360controller import Xbox360Controller
import numpy as np
import time
import cProfile

controller = Xbox360Controller(axis_threshold=0.2)

def get_box():
    data = np.zeros(5, dtype=np.float64)
    lx = controller.axis_l.x
    ly = controller.axis_l.y
    if abs(lx) > abs(ly):
        data[0] = lx
    else:
        data[1] = ly

    rx = controller.axis_r.x
    ry = controller.axis_r.y
    
    if abs(rx) > abs(ry):
        data[3] = rx
    else:
        data[2] = ry

    tl = controller.trigger_l._value
    tr = controller.trigger_r._value
    if abs(tl) > abs(tr):
        data[4] = -tl
    else:
        data[4] = tr
    data = np.around(data, 1)
    print(data)
    return data

idx = 0
while True:
    print(get_box())
    if idx == 200:
        cProfile.run('get_box()', filename='xbox.dat')
    idx += 1

    time.sleep(0.01)
