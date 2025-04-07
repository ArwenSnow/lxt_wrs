import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf
import robot_sim.robots.gofa5.gofa5 as gf5
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# finger_type
finger_type = 'a'

# hnd_type
g = 'l'

# # robot
# rbt = gf5.GOFA5()
# rbt.gen_meshmodel().attach_to(base)
# table = rbt.base_stand.lnks[0]['collision_model']

# finger_s
if finger_type == 'a':
    finger = "objects/finger_a_old.stl"
elif finger_type == 'b':
    finger = "objects/finger_b_old.stl"
elif finger_type == 'c':
    finger = "objects/finger_c_old.stl"
else:
    raise ValueError(f"Invalid finger type selected: {finger_type}")

finger_1 = cm.CollisionModel(finger)
finger_1.set_rgba([.7, .7, .7, 1])
finger_2 = cm.CollisionModel(finger)
finger_2.set_rgba([.7, .7, .7, 1])

# hnd_s
if g == 'l':
    gripper = rf.reconfgripper().lft
elif g == 'r':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()
gripper_b = rf.reconfgripper().body

# tool for planning finger grasps
tool_a = cm.CollisionModel("objects/tool_a.stl")
tool_b = cm.CollisionModel("objects/tool_b_c.stl")
tool_c = cm.CollisionModel("objects/tool_b_c.stl")

# object_s
object_1 = cm.CollisionModel("objects/screw.stl")
object_1.set_rgba([.9, .75, .35, 1])
object_1.set_pos(np.array([.5, .2, -0.015]))

object_2 = cm.CollisionModel("objects/bigbox.stl")
object_2.set_rgba([.9, .75, .35, 1])
object_2.set_pos(np.array([.0, .0, 0.]))

object_3 = cm.CollisionModel("objects/cylinder.stl")
object_3.set_rgba([.9, .75, .35, 1])

if finger_type == 'a':
    tool = tool_a
    target_object = object_1
    target_object.attach_to(base)
elif finger_type == 'b':
    tool = tool_b
    target_object = object_2
    target_object.attach_to(base)
elif finger_type == 'c':
    tool = tool_c
    target_object = object_3
    target_object.attach_to(base)

# flag
found_0 = None
found_1 = None
found_2 = None

# goal_pose
goal_pos = None
goal_rotmat = None

# gripper_grasp_finger
grasp_info_list = gpa.plan_grasps(gripper, tool,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=1, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

finger_grasp_info_list = []
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x, y, z = jaw_center_pos
    x_axis_direction = jaw_center_rotmat[:, 0]
    desired_direction = np.array([-1, 0, 0])
    cos_angle = np.dot(x_axis_direction, desired_direction) / (
            np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction))
    if (g == 'l' and cos_angle == -1) or (g == 'r' and cos_angle == 1):
        finger_grasp_info_list.append(grasp_info)



grasp_info = finger_grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info

# update jawwidth_rng and jaw_center_pos
gripper_b.set_finger_type(finger_type, jaw_center_pos, jaw_center_rotmat, g)
m = gripper_b.jaw_center_pos
n = gripper_b.jawwidth_rng

# finger_grasp_object
object_grasp_info_list = gpa.plan_grasps(gripper_b, target_object,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)

new_object_grasp_info_list = []
for object_grasp_info in object_grasp_info_list:
    m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
    x, y, z = m_jaw_center_pos
    if finger_type == 'a':
        new_object_grasp_info_list.append(object_grasp_info)
    if finger_type == 'b':
        new_object_grasp_info_list.append(object_grasp_info)
    if finger_type == 'c':
        if z <= -1e-08 or z >= 1e-08:
            new_object_grasp_info_list.append(object_grasp_info)

new_object_grasp_info = new_object_grasp_info_list[0]
m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = new_object_grasp_info

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
    gripper_m.lg_jaw_to(jaw_width)
    gripper_m.rg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.132])  # eef根-小夹爪零点（eef坐标系下）
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)  # eef坐标系 → 世坐标系
elif g =='r':
    gripper_m.lg_jaw_to(jaw_width)
    gripper_m.rg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.132])
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)

gripper_m.fix_to(m_pos, m_rotmat)
gripper_m.mg_jaw_to(m_jaw_width)
# gripper_m.gen_meshmodel().attach_to(base)

# 放另一根手指，并打印实时抓取中心pose
if g == 'l':
    gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
    finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
    finger_2.set_pos(finger_2_pos)
    finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
    finger_2.set_rotmat(finger_2_rotmat)
    # finger_1.attach_to(base)
    # finger_2.attach_to(base)
elif g =='r':
    gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
    finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
    finger_2.set_pos(finger_2_pos)
    finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
    finger_2.set_rotmat(finger_2_rotmat)
    # finger_1.attach_to(base)
    # finger_2.attach_to(base)
gripper_m.gripper_bool(g='l')
gripper_m.gripper_bool(g='r')
gripper_m.gripper_bool(g='m')

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

finger_2_pos_new = np.dot(c, finger_2_pos - m_pos) + a
finger_2_rotmat_new = np.dot(c, finger_2_rotmat)
finger_2_new.set_pos(finger_2_pos_new)
finger_2_new.set_rotmat(finger_2_rotmat_new)
objcm_2 = cm.CollisionModel(finger_2_new, cdprimit_type='box', expand_radius=.1)
finger_1_new.attach_to(base)
finger_2_new.attach_to(base)
gripper_m.gen_meshmodel().attach_to(base)

# if not target_object.is_mcdwith([finger_2_new, finger_1_new, rbt]):
#     finger_1_new.attach_to(base)
#     finger_2_new.attach_to(base)
#     gripper_m.gen_meshmodel().attach_to(base)




# for grasp_info in finger_grasp_info_list:
#     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
#
#     # update jawwidth_rng and jaw_center_pos
#     gripper_b.set_finger_type(finger_type, jaw_center_pos, jaw_center_rotmat, g)
#     m = gripper_b.jaw_center_pos
#     n = gripper_b.jawwidth_rng
#
#     # finger_grasp_object
#     object_grasp_info_list = gpa.plan_grasps(gripper_b, target_object,
#                                              angle_between_contact_normals=math.radians(160),
#                                              openning_direction='loc_x',
#                                              max_samples=10, min_dist_between_sampled_contact_points=.005,
#                                              contact_offset=.001)
#
#     new_object_grasp_info_list = []
#     for object_grasp_info in object_grasp_info_list:
#         m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
#         x, y, z = m_jaw_center_pos
#         if finger_type == 'a':
#             new_object_grasp_info_list.append(object_grasp_info)
#         if finger_type == 'b':
#             if m_jaw_width-.237 > 0:
#                 new_object_grasp_info_list.append(object_grasp_info)
#             else:
#                 print(m_jaw_width)
#         if finger_type == 'c':
#             if z <= -1e-08 or z >= 1e-08:
#                 new_object_grasp_info_list.append(object_grasp_info)
#
#     for new_object_grasp_info in new_object_grasp_info_list:
#         m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = new_object_grasp_info
#
#         finger_1_new = cm.CollisionModel(finger)
#         finger_1_new.set_rgba([.7, .7, .7, 1])
#         finger_2_new = cm.CollisionModel(finger)
#         finger_2_new.set_rgba([.7, .7, .7, 1])
#
#         if finger_type == 'a':
#             m_jaw_width = m_jaw_width - .00287
#         elif finger_type == 'b':
#             m_jaw_width = m_jaw_width - .237
#         elif finger_type == 'c':
#             m_jaw_width = m_jaw_width - .03959
#
#         if g == 'l':
#             gripper_m.lg_jaw_to(jaw_width)
#             m_rotmat = hnd_rotmat
#             m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.132])
#             m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#         elif g == 'r':
#             gripper_m.rg_jaw_to(jaw_width)
#             m_rotmat = hnd_rotmat
#             m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.132])
#             m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#
#         gripper_m.fix_to(m_pos, m_rotmat)
#         gripper_m.mg_jaw_to(m_jaw_width)
#         # gripper_m.gen_meshmodel().attach_to(base)
#
#         if g == 'l':
#             gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
#             finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#             finger_2.set_pos(finger_2_pos)
#             finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#             finger_2.set_rotmat(finger_2_rotmat)
#             # finger_1.attach_to(base)
#             # finger_2.attach_to(base)
#         elif g == 'r':
#             gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
#             finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#             finger_2.set_pos(finger_2_pos)
#             finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#             finger_2.set_rotmat(finger_2_rotmat)
#             # finger_1.attach_to(base)
#             # finger_2.attach_to(base)
#         gripper_m.gripper_bool(g='l')
#         gripper_m.gripper_bool(g='r')
#         gripper_m.gripper_bool(g='m')
#
#         gripper_b.grip_at_with_jcpose(m_jaw_center_pos, m_jaw_center_rotmat, m_jaw_width)
#         a = gripper_b.pos
#         b = gripper_b.rotmat
#         gripper_m.fix_to(a, b)
#         gripper_m.mg_jaw_to(m_jaw_width)
#
#         c = np.dot(b, m_rotmat.T)
#         finger_1_pos_new = np.dot(c, np.array([0, 0, 0]) - m_pos) + a
#         finger_1_rotmat_new = c
#         finger_1_new.set_pos(finger_1_pos_new)
#         finger_1_new.set_rotmat(finger_1_rotmat_new)
#
#         finger_2_pos_new = np.dot(c, finger_2_pos - m_pos) + a
#         finger_2_rotmat_new = np.dot(c, finger_2_rotmat)
#         finger_2_new.set_pos(finger_2_pos_new)
#         finger_2_new.set_rotmat(finger_2_rotmat_new)
#
#
#         # finger_1_new.attach_to(base)
#         # finger_2_new.attach_to(base)
#         # gripper_m.gen_meshmodel().attach_to(base)
#
#         if not target_object.is_mcdwith([finger_2_new, finger_1_new]):
#             finger_1_new.attach_to(base)
#             finger_2_new.attach_to(base)
#             gripper_m.gen_meshmodel().attach_to(base)
#             found_1 = True
#             goal_pos = a
#             goal_rotmat = b
#
#         if found_1:
#             # robot_planning
#             try:
#                 start_conf = rbt.get_jnt_values(component_name='arm')
#                 jnt_values = start_conf + np.array([0, 0.3, 0, 0.1, 0.5, 0.5])
#                 rbt.fk(component_name="arm", jnt_values=jnt_values)
#
#                 tgt_pos = rbt.get_gl_tcp("arm")[0]
#                 tgt_rotmat = rbt.get_gl_tcp("arm")[1]
#
#                 goal_pos = goal_pos
#                 goal_rotmat = goal_rotmat
#                 goal_jnt_values = rbt.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
#                 rbt.fk(component_name="arm", jnt_values=goal_jnt_values)
#                 rbt.gen_meshmodel().attach_to(base)
#
#                 rrtc_planner = rrtc.RRTConnect(rbt)
#                 path = rrtc_planner.plan(component_name="arm",
#                                          start_conf=start_conf,
#                                          goal_conf=goal_jnt_values,
#                                          ext_dist=0.05,
#                                          max_time=300)
#                 rbt_current = rbt.get_jnt_values("arm")
#                 print(f"机器人初始关节角度：{start_conf}")
#                 print(f"机器人当前关节角度:{rbt_current}")
#                 found_2 = True
#             except Exception as e:
#                 print(f"Error: Path planning failed with exception: {str(e)}")
#                 found_2 = False
#
#         if found_2:
#             found_0 = True
#             break
#     if found_0:
#         break

base.run()
