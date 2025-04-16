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
# finger_1.attach_to(base)

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

# object_s
object_1 = cm.CollisionModel("objects/screw.stl")
object_1.set_rgba([.9, .75, .35, 1])
object_1.set_pos(np.array([.5, .2, -0.015]))
object_1.attach_to(base)

# gripper_grasp_finger
lfor_grasp_object = gpa.load_pickle_file('lfor_grasp_object', '../robot_con/gofa_con/grasp_reconfgripper/path_list/grasp_finger_info/', 'lfor_grasp_object.pickle')
rfor_grasp_object = gpa.load_pickle_file('rfor_grasp_object', '../robot_con/gofa_con/grasp_reconfgripper/path_list/grasp_finger_info/', 'rfor_grasp_object.pickle')
counter = gpa.load_pickle_file('counter', '../robot_con/gofa_con/grasp_reconfgripper/path_list/grasp_finger_info/', 'counter.pickle')
l_count = counter[0]
r_count = counter[1]
ljaw_width, ljaw_center_pos, ljaw_center_rotmat, lhnd_pos, lhnd_rotmat = lfor_grasp_object[l_count]
rjaw_width, rjaw_center_pos, rjaw_center_rotmat, rhnd_pos, rhnd_rotmat = rfor_grasp_object[r_count]

# update jawwidth_rng and jaw_center_pos
gripper_b.set_finger_type(finger_type, ljaw_center_pos, ljaw_center_rotmat, g)
m = gripper_b.jaw_center_pos
n = gripper_b.jawwidth_rng

# finger_grasp_object
object_grasp_info_list = gpa.plan_grasps(gripper_b, object_1,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)

# new_object_grasp_info_list = []
# for object_grasp_info in object_grasp_info_list:
#     m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
#     x, y, z = m_jaw_center_pos
#     if finger_type == 'a':
#         new_object_grasp_info_list.append(object_grasp_info)
#     if finger_type == 'b':
#         new_object_grasp_info_list.append(object_grasp_info)
#     if finger_type == 'c':
#         if z <= -1e-08 or z >= 1e-08:
#             new_object_grasp_info_list.append(object_grasp_info)
#
# new_object_grasp_info = new_object_grasp_info_list[0]
object_grasp_info = object_grasp_info_list[0]
m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info

finger_1_new = cm.CollisionModel(finger)
finger_1_new.set_rgba([.7, .7, .7, 1])
finger_2_new = cm.CollisionModel(finger)
finger_2_new.set_rgba([.7, .7, .7, 1])

if finger_type == 'a':
    m_jaw_width = m_jaw_width - .00287
elif finger_type == 'b':
    m_jaw_width = m_jaw_width - .237
elif finger_type == 'c':
    m_jaw_width = m_jaw_width - .03959

if g == 'l':
    gripper_m.lg_jaw_to(ljaw_width)
    gripper_m.rg_jaw_to(ljaw_width)
    m_rotmat = lhnd_rotmat
    m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.132])  # eef根-小夹爪零点（eef坐标系下）
    m_pos = lhnd_pos + lhnd_rotmat.dot(m_pos)  # eef坐标系 → 世坐标系
elif g =='r':
    gripper_m.lg_jaw_to(ljaw_width)
    gripper_m.rg_jaw_to(ljaw_width)
    m_rotmat = lhnd_rotmat
    m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.132])
    m_pos = lhnd_pos + lhnd_rotmat.dot(m_pos)

gripper_m.fix_to(m_pos, m_rotmat)
gripper_m.mg_jaw_to(m_jaw_width)
# gripper_m.gen_meshmodel().attach_to(base)

# # 放另一根手指，并打印实时抓取中心pose
# if g == 'l':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
#     finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#     finger_2.set_pos(finger_2_pos)
#     finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#     finger_2.set_rotmat(finger_2_rotmat)
#     # finger_1.attach_to(base)
#     # finger_2.attach_to(base)
# elif g =='r':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
#     finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#     finger_2.set_pos(finger_2_pos)
#     finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#     finger_2.set_rotmat(finger_2_rotmat)
#     # finger_1.attach_to(base)
#     # finger_2.attach_to(base)
# gripper_m.gripper_bool(g='l')
# gripper_m.gripper_bool(g='r')
# gripper_m.gripper_bool(g='m')

# gripper_m放最终的位置
gripper_b.grip_at_with_jcpose(m_jaw_center_pos, m_jaw_center_rotmat, m_jaw_width)
a = gripper_b.pos
b = gripper_b.rotmat
gripper_m.fix_to(a, b)
gripper_m.mg_jaw_to(m_jaw_width)

c = np.dot(b, m_rotmat.T)
finger_1_pos_new = np.dot(c, np.array([0, 0, 0]) - m_pos) + a  # 第一次规划抓取手指时 → 世界坐标系
finger_1_rotmat_new = c
finger_1_new.set_pos(finger_1_pos_new)
finger_1_new.set_rotmat(finger_1_rotmat_new)
objcm_1 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)

finger_2_pos_new = np.dot(c, rjaw_center_pos - m_pos) + a
finger_2_rotmat_new = np.dot(c, rjaw_center_rotmat)
finger_2_new.set_pos(finger_2_pos_new)
finger_2_new.set_rotmat(finger_2_rotmat_new)
objcm_2 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)
finger_1_new.attach_to(base)
finger_2_new.attach_to(base)
gripper_m.gen_meshmodel().attach_to(base)

base.run()
