import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf
import robot_sim.robots.gofa5.gofa5 as gf5

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# finger_type
finger_type = 'a'

# hnd_type
g = 'l'

# finger_s
if finger_type == 'a':
    finger = cm.CollisionModel("objects/finger_a.stl")
elif finger_type == 'b':
    finger = "objects/finger_b_old.stl"
elif finger_type == 'c':
    finger = "objects/finger_c_old.stl"
else:
    raise ValueError(f"Invalid finger type selected: {finger_type}")

finger_1 = cm.CollisionModel(finger)
finger_1.set_pos(np.array([0, 0, 0]))
finger_1.set_rgba([.7, .7, .7, 1])
finger_1.attach_to(base)

finger_2 = cm.CollisionModel(finger)
finger_2.set_rgba([.7, .7, .7, 1])
# finger_2.attach_to(base)

# hnd_s
if g == 'l':
    gripper = rf.reconfgripper().lft
elif g == 'r':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()
gripper_b = rf.reconfgripper().body

# tool_s
tool = cm.CollisionModel("objects/tool.stl")
tool.set_pos(np.array([0, 0, 0]))

# gripper_grasp_finger
grasp_info_list = gpa.plan_grasps(gripper, tool,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=100, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('finger', grasp_info_list, './', 'reconfgripper_finger_grasps.pickle')
grasp_info_list = gpa.load_pickle_file('finger', './', 'reconfgripper_finger_grasps.pickle')

grasp_info = grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
print(jaw_center_pos)
print(jaw_center_rotmat)
gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
gripper.gen_meshmodel(rgba=[0, 0, 1, 1]).attach_to(base)

mg_jawwidth = 0.05
if g == 'l':
    gripper_m.lg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    value = np.array([-.053 - mg_jawwidth / 2, .018, -.132])  # dh60根部和小夹爪根部的pos差值
    m_pos = hnd_pos + hnd_rotmat.dot(value)                    # 得到世界坐标系下的dh60根部pos
elif g == 'r':
    gripper_m.rg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    value = np.array([.053 + mg_jawwidth / 2, -.018, -.132])
    m_pos = hnd_pos + hnd_rotmat.dot(value)

gripper_m.fix_to(m_pos, m_rotmat)
gripper_m.mg_jaw_to(mg_jawwidth)
gripper_m.gen_meshmodel().attach_to(base)

# for grasp_info in grasp_info_list:
#     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
#     gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
#
#     mg_jawwidth = 0.05
#     if g == 'l':
#         gripper_m.lg_jaw_to(jaw_width)
#         m_rotmat = hnd_rotmat
#         value = np.array([-.0535 - mg_jawwidth / 2, .018, -.132])  # dh60根部和小夹爪根部的pos差值
#         m_pos = hnd_pos + hnd_rotmat.dot(value)  # 得到世界坐标系下的dh60根部pos
#     elif g == 'r':
#         gripper_m.rg_jaw_to(jaw_width)
#         m_rotmat = hnd_rotmat
#         value = np.array([.0535 + mg_jawwidth / 2, -.018, -.132])
#         m_pos = hnd_pos + hnd_rotmat.dot(value)
#
#     gripper_m.fix_to(m_pos, m_rotmat)
#     gripper_m.mg_jaw_to(mg_jawwidth)
#     gripper_m.gen_meshmodel().attach_to(base)

base.run()
