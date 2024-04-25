import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf

base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
gm.gen_frame().attach_to(base)
gripper = rf.reconfgripper()

counter = [0]
flag = [0]
jawwidth_rng = np.linspace(0.0, 0.06, 20)
gripper_mesh = []


def update(jawwidth_rng, counter, flag, task):
    for model in gripper_mesh:
        model.detach()
    gripper_mesh.clear()
    gripper.mg_jaw_to(jawwidth_rng[counter[0]])
    gripper.lft.fix_to(pos=gripper.body.lft.jnts[2]['gl_posq'],
                       rotmat=gripper.body.lft.jnts[2]['gl_rotmatq'])
    gripper.rgt.fix_to(pos=gripper.body.rgt.jnts[2]['gl_posq'],
                       rotmat=gripper.body.rgt.jnts[2]['gl_rotmatq'])
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