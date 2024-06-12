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
object = cm.CollisionModel("objects/box_text.stl")
object.set_rgba([.9, .75, .35, 1])
object.attach_to(base)

# hnd_s
g='lgt'
if g == 'lgt':
    gripper = rf.reconfgripper().lft
elif g =='rgt':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()


grasp_info_list = gpa.plan_grasps(gripper, object,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
gpa.write_pickle_file('holder', grasp_info_list, './', 'cobg_holder_grasps.pickle')
# grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')

a = gripper_m.pos
b = gripper_m.rotmat

grasp_info = grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
gripper.gen_meshmodel().attach_to(base)

# if g == 'lgt':
#     m_rotmat = hnd_rotmat
#     gripper_m.lg_jaw_to(jaw_width)
# elif g =='rgt':
#     m_rotmat = np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi * 1 ),b)
#     m_rotmat = hnd_rotmat.dot(m_rotmat)
#     gripper_m.rg_jaw_to(jaw_width)
# m_pos = np.array([-.0656, 0, -.1373]) + gripper.rotmat.dot(a)
# m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
# gripper_m.fix_to(m_pos, m_rotmat)
# gripper_m.mg_open()
# # gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)
# gripper_m.gen_meshmodel().attach_to(base)
#
# # for grasp_info in grasp_info_list:
# #     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
# #     gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
# #     gripper.gen_meshmodel().attach_to(base)
# #     # gripper.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)
# #
# #     if g == 'lgt':
# #         m_rotmat = hnd_rotmat
# #         gripper_m.lg_jaw_to(jaw_width)
# #     elif g == 'rgt':
# #         m_rotmat = np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi * 1), b)
# #         m_rotmat = hnd_rotmat.dot(m_rotmat)
# #         gripper_m.rg_jaw_to(jaw_width)
# #     m_pos = np.array([-.0626, 0, -.1373]) + gripper.rotmat.dot(a)
# #     m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
# #     gripper_m.fix_to(m_pos, m_rotmat)
# #     gripper_m.mg_open()
# #     gripper_m.gen_meshmodel().attach_to(base)
# #     # gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)

base.run()
