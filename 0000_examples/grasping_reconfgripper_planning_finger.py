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

# finger
finger_a_1 = cm.CollisionModel("objects/finger_a.stl")
finger_a_1.set_rgba([.7, .7, .7, 1])
finger_a_1.attach_to(base)

finger_a_2 = cm.CollisionModel("objects/finger_a.stl")
finger_a_2.set_rgba([.7, .7, .7, 1])

# object
object_1 = cm.CollisionModel("objects/box_text.stl")
object_1.set_rgba([.9, .75, .35, 1])

# hnd_s
g = 'lft'
if g == 'lft':
    gripper = rf.reconfgripper().lft
elif g =='rgt':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()
gripper_b = rf.reconfgripper().body

grasp_info_list = gpa.plan_grasps(gripper, finger_a_1,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

object_grasp_info_list = gpa.plan_grasps(gripper_b, object_1,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)

finger_grasp_info_list = []
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x, y, z = jaw_center_pos
    if -0.0025 <= x <= 0.0025 and -0.015 <= y <= 0.015 and -0.015 <= z <= 0.015:
        x_axis_direction = jaw_center_rotmat[:, 0]
        desired_direction = np.array([-1, 0, 0])
        cos_angle = np.dot(x_axis_direction, desired_direction) / (
                    np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction))
        if (g == 'lft' and cos_angle == -1) or (g == 'rgt' and cos_angle == 1):
            finger_grasp_info_list.append(grasp_info)

# grasp_info = finger_grasp_info_list[0]
# jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
# gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
# gripper.gen_meshmodel().attach_to(base)
#
# object_grasp_info = object_grasp_info_list[0]
# m_jaw_width, m_jaw_center_pos, m_jaw_center_rotmat, m_hnd_pos, m_hnd_rotmat = object_grasp_info
#
# mg_jawwidth = m_jaw_width
# if g == 'lft':
#     gripper_m.lg_jaw_to(jaw_width)
#     m_rotmat = hnd_rotmat
#     m_pos = np.array([-.0535 - mg_jawwidth / 2, .018, -.135])
#     m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
# elif g =='rgt':
#     gripper_m.rg_jaw_to(jaw_width)
#     m_rotmat = hnd_rotmat
#     m_pos = np.array([.0535 + mg_jawwidth / 2, -.018, -.135])
#     m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#
# gripper_m.fix_to(m_pos, m_rotmat)
# gripper_m.mg_jaw_to(mg_jawwidth)
# gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)
#
# if g == 'lft':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="l")
#     finger_a_2.set_pos(np.array([-0.054 - mg_jawwidth, 0, 0]))
#     rgt_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#     finger_a_2.set_rotmat(rgt_rotmat)
#     finger_a_2.attach_to(base)
# elif g =='rgt':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="r")
#     finger_a_2.set_pos(np.array([-0.054 - mg_jawwidth, 0, 0]))
#     lft_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
#     finger_a_2.set_rotmat(lft_rotmat)
#     finger_a_2.attach_to(base)
# gripper_m.gripper_bool(g='l')
# gripper_m.gripper_bool(g='r')
# gripper_m.gripper_bool(g='m')
#
# gripper_b.fix_to(m_pos, m_rotmat)
# gripper_b.gen_meshmodel().attach_to(base)
#
# object_1_pos = -m_pos + m_rotmat.dot(object_1.get_pos() - m_hnd_pos)
# object_1_rotmat = np.dot(m_rotmat, np.dot(object_1.get_rotmat(), m_rotmat.T))
# object_1.set_pos(object_1_pos)
# # object_1.set_rotmat(object_1_rotmat)
# # object_1.attach_to(base)

for grasp_info in finger_grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper.gen_meshmodel().attach_to(base)

    mg_jawwidth = 0.062
    if g == 'lft':
        gripper_m.lg_jaw_to(jaw_width)
        m_rotmat = hnd_rotmat
        m_pos = np.array([-.0535 - mg_jawwidth / 2, .018, -.135])
        m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
    elif g == 'rgt':
        gripper_m.rg_jaw_to(jaw_width)
        m_rotmat = hnd_rotmat
        m_pos = np.array([.0535 + mg_jawwidth / 2, -.018, -.135])
        m_pos = hnd_pos + hnd_rotmat.dot(m_pos)

    gripper_m.fix_to(m_pos, m_rotmat)
    gripper_m.mg_jaw_to(mg_jawwidth)
    gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)

    if g == 'lft':
        gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="l")
        finger_a_2.set_pos(np.array([-0.054 - mg_jawwidth, 0, 0]))
        rgt_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
        finger_a_2.set_rotmat(rgt_rotmat)
        finger_a_2.attach_to(base)
    elif g == 'rgt':
        gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="r")
        finger_a_2.set_pos(np.array([-0.054 - mg_jawwidth, 0, 0]))
        lft_rotmat = np.dot(np.eye(3), rm.rotmat_from_euler(0, 0, math.pi))
        finger_a_2.set_rotmat(lft_rotmat)
        finger_a_2.attach_to(base)
    gripper_m.gripper_bool(g='l')
    gripper_m.gripper_bool(g='r')
    gripper_m.gripper_bool(g='m')

base.run()
