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

# object
object_1 = cm.CollisionModel("objects/box_text.stl")
object_1.set_rgba([.9, .75, .35, 1])
object_1.attach_to(base)

object_2 = cm.CollisionModel("objects/ball.stl")
object_2.set_rgba([.9, .75, .35, 1])
# object_2.attach_to(base)

# hnd_s
g = 'rgt'
if g == 'lft':
    gripper = rf.reconfgripper().lft
elif g =='rgt':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()

grasp_info_list = gpa.plan_grasps(gripper, object_1,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')


grasp_info = grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
gripper.gen_meshmodel().attach_to(base)

mg_jawwidth = 0.062
if g == 'lft':
    gripper_m.lg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    m_pos = np.array([-.0535 - mg_jawwidth / 2, .018, -.132])
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
elif g =='rgt':
    gripper_m.rg_jaw_to(jaw_width)
    m_rotmat = hnd_rotmat
    m_pos = np.array([.0535 + mg_jawwidth / 2, -.018, -.132])
    m_pos = hnd_pos + hnd_rotmat.dot(m_pos)

gripper_m.fix_to(m_pos, m_rotmat)
gripper_m.mg_jaw_to(mg_jawwidth)
gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)

# if g == 'lft':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="l")
# elif g =='rgt':
#     gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="r")
# gripper_m.gripper_bool(g='l')
# gripper_m.gripper_bool(g='r')
# gripper_m.gripper_bool(g='m')


# for grasp_info in grasp_info_list:
#     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
#     gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
#     gripper.gen_meshmodel().attach_to(base)
#
#     mg_jawwidth = 0.062
#     if g == 'lft':
#         gripper_m.lg_jaw_to(jaw_width)
#         m_rotmat = hnd_rotmat
#         m_pos = np.array([-.0535 - mg_jawwidth / 2, .018, -.135])
#         m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#     elif g == 'rgt':
#         gripper_m.rg_jaw_to(jaw_width)
#         m_rotmat = hnd_rotmat
#         m_pos = np.array([.0535 + mg_jawwidth / 2, -.018, -.135])
#         m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
#
#     gripper_m.fix_to(m_pos, m_rotmat)
#     gripper_m.mg_jaw_to(mg_jawwidth)
#     gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)
#
#     if g == 'lft':
#         gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="l")
#     elif g == 'rgt':
#         gripper_m.center_pos_global(jaw_center_pos, jaw_center_rotmat, mg_jawwidth, g="r")
#     gripper_m.gripper_bool(g='l')
#     gripper_m.gripper_bool(g='r')
#     gripper_m.gripper_bool(g='m')

base.run()
