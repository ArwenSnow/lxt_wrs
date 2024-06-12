import copy
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.robots.gofa5.gofa5 as gf5

base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
gm.gen_frame().attach_to(base)

rbt_s = gf5.GOFA5()
rbt_s.hnd.open()
rbt_s.gen_meshmodel().attach_to(base)

# object
objcm_name = "box"
obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
obj.set_rgba([.9, .75, .35, 1])
obj.set_pos(np.array([.4,-.2,-.015]))
obj.set_rotmat()
obj.attach_to(base)

# object_goal
obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
obj_goal.set_rgba([1, 1, 1, 1])
obj_goal.set_pos(np.array([.3,.4,-.015]))
obj.set_rotmat()
obj_goal.attach_to(base)

gripper_s = dh.Dh60()
grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    pos = gripper_s.pos + np.array([.4, -.2, -.015])
    gripper_s.fix_to(pos=pos,rotmat=hnd_rotmat)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    pos = gripper_s.pos + np.array(([.3, .4, -.015]))
    gripper_s.fix_to(pos=pos,rotmat=hnd_rotmat)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)

grasp_info = grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
start_pos = jaw_center_pos + np.array([.4, -.2, -.015])
start_rotmat = hnd_rotmat
goal_jnt_values = rbt_s.ik(tgt_pos=start_pos, tgt_rotmat=start_rotmat)
rbt_s.fk(component_name="arm", jnt_values=goal_jnt_values)
rbt_s.gen_meshmodel().attach_to(base)

goal_pos = jaw_center_pos + np.array([.3, .4, -.015])
goal_rotmat = hnd_rotmat
goal_jnt_values = rbt_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
rbt_s.fk(component_name="arm", jnt_values=goal_jnt_values)
rbt_s.gen_meshmodel().attach_to(base)

base.run()