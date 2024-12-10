import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rf

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# finger_type
finger_type = 'a'

# hnd_type
g = 'lft'

# finger
finger_1 = cm.CollisionModel("objects/finger_a.stl")
finger_1.set_rgba([.7, .7, .7, 1])
# finger_1.attach_to(base)
finger_2 = cm.CollisionModel("objects/finger_a.stl")
finger_2.set_rgba([.7, .7, .7, 1])

finger_1_new = cm.CollisionModel("objects/finger_a.stl")
finger_1_new.set_rgba([.7, .7, .7, 1])
finger_2_new = cm.CollisionModel("objects/finger_a.stl")
finger_2_new.set_rgba([.7, .7, .7, 1])

# hnd_s
if g == 'lft':
    gripper = rf.reconfgripper().lft
elif g =='rgt':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()
gripper_b = rf.reconfgripper().body

# object_s
object_1 = cm.CollisionModel("objects/screw.stl")
object_1.set_rgba([.9, .75, .35, 1])
object_1.attach_to(base)

# gripper_grasp_finger
grasp_info_list = gpa.plan_grasps(gripper, finger_1,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=100, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

finger_grasp_info_list = []
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x, y, z = jaw_center_pos
    if x == 0 and -0.015 <= y <= 0.015 and -0.015 <= z <= 0.015:
        x_axis_direction = jaw_center_rotmat[:, 0]
        desired_direction = np.array([-1, 0, 0])
        cos_angle = np.dot(x_axis_direction, desired_direction) / (
                    np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction))
        if (g == 'lft' and cos_angle == -1) or (g == 'rgt' and cos_angle == 1):
            finger_grasp_info_list.append(grasp_info)

grasp_info = finger_grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info

# update jawwidth_rng and jaw_center_pos
gripper_b.set_finger_type(finger_type, jaw_center_pos, jaw_center_rotmat, g='l')
m = gripper_b.jaw_center_pos
n = gripper_b.jawwidth_rng

# finger_grasp_object
object_grasp_info_list = gpa.plan_grasps(gripper_b, object_1,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)

object_grasp_info = object_grasp_info_list[0]
m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info

if finger_type == 'a':
    m_jaw_width = m_jaw_width - .00287
elif finger_type == 'b':
    m_jaw_width = m_jaw_width - .237
elif finger_type == 'c':
    m_jaw_width = m_jaw_width - .04451

if g == 'lft':
    gripper_m.lg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.135])
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
elif g =='rgt':
    gripper_m.rg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.135])
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)

gripper_m.fix_to(m_pos, m_rotmat)
# gripper_m.mg_jaw_to(m_jaw_width)
# gripper_m.gen_meshmodel().attach_to(base)

if g == 'lft':
    gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
    finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
    finger_2.set_pos(finger_2_pos)
    finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
    finger_2.set_rotmat(finger_2_rotmat)
    # finger_2.attach_to(base)
elif g =='rgt':
    gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
    finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
    finger_2.set_pos(finger_2_pos)
    finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
    finger_2.set_rotmat(finger_2_rotmat)
    # finger_2.attach_to(base)
gripper_m.gripper_bool(g='l')
gripper_m.gripper_bool(g='r')
gripper_m.gripper_bool(g='m')

gripper_b.grip_at_with_jcpose(m_jaw_center_pos, m_jaw_center_rotmat, m_jaw_width)
a = gripper_b.pos
b = gripper_b.rotmat
gripper_m.fix_to(a, b)
gripper_m.mg_jaw_to(m_jaw_width)
gripper_m.gen_meshmodel().attach_to(base)

c = np.dot(b, m_rotmat.T)
finger_1_pos_new = np.dot(c, np.array([0, 0, 0]) - m_pos) + a
finger_1_rotmat_new = c
finger_1_new.set_pos(finger_1_pos_new)
finger_1_new.set_rotmat(finger_1_rotmat_new)
finger_1_new.attach_to(base)

finger_2_pos_new = np.dot(c, finger_2_pos - m_pos) + a
finger_2_rotmat_new = np.dot(c, finger_2_rotmat)
finger_2_new.set_pos(finger_2_pos_new)
finger_2_new.set_rotmat(finger_2_rotmat_new)
finger_2_new.attach_to(base)
print(m)
print(n)
print(m_jaw_width)

# for object_grasp_info in object_grasp_info_list:
#     m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
#     if g == 'lft':
#         gripper_m.lg_jaw_to(jaw_width)
#         m_rotmat = hnd_rotmat
#         m_pos = np.array([-.053 - m_jaw_width / 2, .018, -.135])
#         m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#     elif g == 'rgt':
#         gripper_m.rg_jaw_to(jaw_width)
#         m_rotmat = hnd_rotmat
#         m_pos = np.array([.053 + m_jaw_width / 2, -.018, -.135])
#         m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#
#     gripper_m.fix_to(m_pos, m_rotmat)
#     gripper_m.mg_jaw_to(m_jaw_width)
#     # gripper_m.gen_meshmodel().attach_to(base)
#
#     if g == 'lft':
#         gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="l")
#         finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#         finger_2.set_pos(finger_2_pos)
#         finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#         finger_2.set_rotmat(finger_2_rotmat)
#         # finger_2.attach_to(base)
#     elif g == 'rgt':
#         gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, m_jaw_width, g="r")
#         finger_2_pos = np.array([-0.053 - m_jaw_width, 0, 0])
#         finger_2.set_pos(finger_2_pos)
#         finger_2_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#         finger_2.set_rotmat(finger_2_rotmat)
#         # finger_2.attach_to(base)
#     gripper_m.gripper_bool(g='l')
#     gripper_m.gripper_bool(g='r')
#     gripper_m.gripper_bool(g='m')
#
#     gripper_b.grip_at_with_jcpose(m_jaw_center_pos, m_jaw_center_rotmat, m_jaw_width)
#     a = gripper_b.pos
#     b = gripper_b.rotmat
#     gripper_m.fix_to(a, b)
#     gripper_m.mg_jaw_to(m_jaw_width)
#     gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)
#
#     finger_1_new = cm.CollisionModel("objects/finger_a.stl")
#     finger_1_new.set_rgba([.7, .7, .7, 1])
#     finger_2_new = cm.CollisionModel("objects/finger_a.stl")
#     finger_2_new.set_rgba([.7, .7, .7, 1])
#     c = np.dot(b, m_rotmat.T)
#     finger_1_pos_new = np.dot(c, np.array([0, 0, 0]) - m_pos) + a
#     finger_1_rotmat_new = c
#     finger_1_new.set_pos(finger_1_pos_new)
#     finger_1_new.set_rotmat(finger_1_rotmat_new)
#     finger_1_new.attach_to(base)
#
#     finger_2_pos_new = np.dot(c, finger_2_pos - m_pos) + a
#     finger_2_rotmat_new = np.dot(c, finger_2_rotmat)
#     finger_2_new.set_pos(finger_2_pos_new)
#     finger_2_new.set_rotmat(finger_2_rotmat_new)
#     finger_2_new.attach_to(base)


base.run()
