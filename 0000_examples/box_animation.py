import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta as cbt
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
gm.gen_frame().attach_to(base)
box = cm.gen_box([0.1,0.2,0.3])

pos_list = np.linspace([0,0,0], [.5,0,0], 5)
rot_list = [np.eye(3) for item in pos_list]
box_conf_list = [rm.homomat_from_posrot(pos_list[i], rot_list[i]) for i in range(len(pos_list))]

counter = [0]
flag = [0]

def update(box_conf_list, counter, flag, task):
    # box = cm.gen_box([0.1, 0.2, 0.3])
    box.set_homomat(box_conf_list[counter[0]])
    box.attach_to(base)
    if flag[0]  == 0:
        counter[0]+=1
        if counter[0] == len(box_conf_list)-1:
            flag[0] = 1
    else:
        counter[0] = counter[0]-1
        if counter[0] == 0:
            flag[0]  = 0

    return task.again

taskMgr.doMethodLater(0.5, update, "update",
                      extraArgs=[box_conf_list, counter, flag],
                      appendTask=True)


base.run()