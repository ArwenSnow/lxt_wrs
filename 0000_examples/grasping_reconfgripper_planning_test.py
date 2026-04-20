import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper_old as rf
import robot_sim.robots.gofa5.gofa5 as gf5

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
# gm.gen_frame().attach_to(base)

# finger_type
finger_type = 'd'

# hnd_type
g = 'l'

# finger_s
if finger_type == 'a':
    finger = cm.CollisionModel("objects/finger_a.stl")
elif finger_type == 'b':
    finger = cm.CollisionModel("objects/finger_b_1.0.stl")
elif finger_type == 'c':
    finger = cm.CollisionModel("objects/finger_b_2.0.stl")
elif finger_type == 'd':
    finger = cm.CollisionModel("objects/finger_b_3.0.stl")
elif finger_type == 'e':
    finger = cm.CollisionModel("objects/finger_c_2.0.stl")
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
    gripper = rf.Reconfgripper().lft
elif g == 'r':
    gripper = rf.Reconfgripper().rgt
gripper_m = rf.Reconfgripper()
gripper_b = rf.Reconfgripper().body

# tool_s
tool = cm.CollisionModel("objects/tool.stl")
tool.set_pos(np.array([0, 0, 0]))

# gripper_grasp_finger
grasp_info_list = gpa.plan_grasps(gripper, tool,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=10, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('finger', grasp_info_list, './', 'reconfgripper_finger_grasps.pickle')
grasp_info_list = gpa.load_pickle_file('finger', './', 'reconfgripper_finger_grasps.pickle')

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

grasp_info = finger_grasp_info_list[5]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
print(jaw_center_pos)
print(jaw_center_rotmat)
gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
gripper.gen_meshmodel().attach_to(base)

mg_jawwidth = 0.05
if g == 'l':
    gripper_m.lg_jaw_to(jaw_width)
    gripper_m.rg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    value = np.array([-.053 - mg_jawwidth / 2, .018, -.132])  # dh60根部和小夹爪根部的pos差值
    m_pos = hnd_pos + hnd_rotmat.dot(value)                    # 得到世界坐标系下的dh60根部pos
elif g == 'r':
    gripper_m.rg_jaw_to(jaw_width)
    gripper_m.lg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    value = np.array([.053 + mg_jawwidth / 2, -.018, -.132])
    m_pos = hnd_pos + hnd_rotmat.dot(value)

if g == 'l':
    gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="l")
    finger_2_pos = np.array([-0.053 - mg_jawwidth, 0, 0])
    finger_2.set_pos(finger_2_pos)
    finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
    finger_2.set_rotmat(finger_2_rotmat)
    finger_1.attach_to(base)
    finger_2.attach_to(base)
elif g == 'r':
    gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="r")
    finger_2_pos = np.array([-0.053 - mg_jawwidth, 0, 0])
    finger_2.set_pos(finger_2_pos)
    finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
    finger_2.set_rotmat(finger_2_rotmat)
    finger_1.attach_to(base)
    finger_2.attach_to(base)

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
#     if g == 'l':
#         gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="l")
#         finger_2_pos = np.array([-0.053 - mg_jawwidth, 0, 0])
#         finger_2.set_pos(finger_2_pos)
#         finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#         finger_2.set_rotmat(finger_2_rotmat)
#         finger_1.attach_to(base)
#         finger_2.attach_to(base)
#     elif g == 'r':
#         gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="r")
#         finger_2_pos = np.array([-0.053 - mg_jawwidth, 0, 0])
#         finger_2.set_pos(finger_2_pos)
#         finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#         finger_2.set_rotmat(finger_2_rotmat)
#         finger_1.attach_to(base)
#         finger_2.attach_to(base)
#
#     gripper_m.fix_to(m_pos, m_rotmat)
#     gripper_m.mg_jaw_to(mg_jawwidth)
#     gripper_m.gen_meshmodel(rgba=[0, 1, 0, .1]).attach_to(base)

base.run()
