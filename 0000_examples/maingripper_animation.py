import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.maingripper.maingripper as mg

base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
gm.gen_frame().attach_to(base)
gripper = mg.maingripper()

counter = [0]
flag = [0]
jawwidth_rng = np.linspace(0.0, 0.06, 10)
gripper_mesh = []


def update(jawwidth_rng, counter, flag, task):
    for model in gripper_mesh:
        model.detach()
    gripper_mesh.clear()
    gripper.jaw_to(jawwidth_rng[counter[0]])
    gripper_mesh_model = gripper.gen_meshmodel()
    gripper_mesh_model.attach_to(base)
    gripper_mesh.append(gripper_mesh_model)

    if flag[0] == 0:
        counter[0] += 1
        if counter[0] == len(jawwidth_rng) - 1:
            flag[0] = 1
    else:
        counter[0] -= 1
        if counter[0] == 0:
            flag[0] = 0

    return task.again


taskMgr.doMethodLater(0.2, update, "update",
                      extraArgs=[jawwidth_rng, counter, flag],
                      appendTask=True)

base.run()