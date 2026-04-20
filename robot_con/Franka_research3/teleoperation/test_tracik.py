import numpy as np

import robot_sim.robots.Franka_research3.Franka_research3 as fr3
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math


base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
gm.gen_frame().attach_to(base)

robot_s = fr3.Franka_research3()
robot_s.gen_meshmodel().attach_to(base)     # 原始

fr3_pos = np.array([.6, 0, 0])
fr3_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi/2)
robot_s.fix_to(fr3_pos, fr3_rot)
robot_s.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)   # 绿色是挪位置后的

test_pos, test_rot = robot_s.get_gl_tcp("arm")
tgt_pos = test_pos.copy()
tgt_pos[0] += .1
tgt_pos[1] += .1
tgt_pos[2] -= .2
tgt_rotmat = test_rot

tgt_pos_r = fr3_rot.T.dot(tgt_pos - fr3_pos)
tgt_rotmat_r = fr3_rot.T.dot(tgt_rotmat)
seed = robot_s.get_jnt_values("arm")
print(seed)
conf = robot_s.tracik(tgt_pos=tgt_pos_r, tgt_rotmat=tgt_rotmat_r, seed_jnt_values=seed)
print("conf=", conf)
robot_s.fk("arm", conf)
robot_s.gen_meshmodel(rgba=[1, 0, 0, 1]).attach_to(base)     # 红色的是挪位置后的tracik

conf_numik = robot_s.ik("arm", tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
robot_s.fk("arm", conf_numik)
robot_s.gen_meshmodel(rgba=[0, 0, 1, 1]).attach_to(base)

base.run()