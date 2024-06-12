##reconfgripper grasps the object through its fingers
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
lft_finger = cm.CollisionModel("objects/finger.stl")
lft_finger.set_rgba([.6, .6, .6, 1])
lft_finger.attach_to(base)

rgt_finger = cm.CollisionModel("objects/finger.stl")
rgt_finger.set_rotmat(np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi * 1), lft_finger.get_rotmat()))
rgt_finger.set_rgba([.6, .6, .6, 1])

object = cm.CollisionModel("objects/nut.stl")
object.set_rgba([.65, .65, .65, 1])
object.attach_to(base)

# hnd_s
gripper_l= rf.reconfgripper().lft
gripper_r= rf.reconfgripper().rgt
gripper_m = rf.reconfgripper().body

a=gripper_m.pos
b=gripper_m.rotmat

# grasp_info
lft_jaw_center_pos = np.zeros(3)
lft_jaw_center_rotmat = np.eye(3)
c = lft_jaw_center_rotmat
mg_jawwidth = .076
contact_offset = .0025
jaw_width = .008 + contact_offset*2
angle_increment = math.pi*45/180
grasp_info_list = []


grasp_info_list = gpa.plan_grasps(gripper_m, object,
                                  angle_between_contact_normals=math.radians(160),
                                  openning_direction='loc_x',
                                  max_samples=5, min_dist_between_sampled_contact_points=.005,
                                  contact_offset=.001)
grasp_info = grasp_info_list[0]
jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
# lft_finger.set_rotmat(jaw_center_rotmat)
l = jaw_center_pos+np.array([jaw_width/2+.02505, 0, -.04957])
lft_pos = l + np.dot(jaw_center_rotmat,lft_jaw_center_pos)
lft_finger.set_pos(lft_pos)
lft_finger.attach_to(base)
rgt_finger.set_pos(jaw_center_pos+np.array([-jaw_width/2-.02505, 0, -.04957]))
rgt_finger.attach_to(base)

l_pos=jaw_center_pos+np.array([jaw_width/2+.02505, 0, -.04957])
l_rotmat = lft_finger.get_rotmat()
gripper_l.grip_at_with_jcpose(l_pos, l_rotmat, jaw_width)
gripper_l.gen_meshmodel().attach_to(base)

r_pos=jaw_center_pos+np.array([-jaw_width/2-.02505, 0, -.04957])
r_rotmat = rgt_finger.get_rotmat()
gripper_r.grip_at_with_jcpose(r_pos, r_rotmat, jaw_width)
gripper_r.gen_meshmodel().attach_to(base)

m_pos = np.array([-.0278, 0, -.19273])+l_pos
# m_pos = hnd_pos + hnd_rotmat.dot(m_pos)
m_rotmat = l_rotmat
gripper_m.fix_to(m_pos, m_rotmat)
gripper_m.mg_jaw_to(jaw_width)
gripper_m.gen_meshmodel().attach_to(base)




base.run()


