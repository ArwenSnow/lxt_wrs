import copy
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh.dh as dh
import robot_sim.robots.GOFA5.GOFA5 as gf5

base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
gm.gen_frame().attach_to(base)

rbt_s = gf5.GOFA5()

rbt_s.hnd.mg_open()
rbt_s.gen_meshmodel().attach_to(base)
objcm_name = "box"
obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
obj.set_rgba([.9, .75, .35, 1])

obj.set_pos()
obj.set_rotmat()
obj.attach_to(base)

obj_goal = copy.deepcopy(obj)
obj_goal.set_pos()
obj.set_rotmat()
obj_goal.attach_to(base)

base.run()
gripper_s = dh.dh()
objcm_name = "box"
obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
obj.set_rgba([.9, .75, .35, 1])
obj.attach_to(base)
grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0,1,0,0.01]).attach_to(base)

base.run()