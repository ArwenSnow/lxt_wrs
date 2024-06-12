#reconfgripper grasps a finger
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
object = cm.CollisionModel("objects/finger.stl")
object.set_rgba([.9, .75, .35, 1])
object.attach_to(base)

# hnd_s
g='rgt'
if g == 'lgt':
    gripper = rf.reconfgripper().lft
elif g =='rgt':
    gripper = rf.reconfgripper().rgt
gripper_m = rf.reconfgripper()

a=gripper_m.pos
b=gripper_m.rotmat

# grasp_info
jaw_center_pos = np.zeros(3)
jaw_center_rotmat = np.eye(3)
contact_offset = .0025
jaw_width = .008 + contact_offset*2
angle_increment = math.pi*45/180


for i in range(8):
    jaw_center_rotmat = np.dot(rm.rotmat_from_axangle([1, 0, 0], i*angle_increment ),jaw_center_rotmat)
    gripper.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)

    if g == 'lgt':
        m_rotmat = jaw_center_rotmat
        gripper_m.lg_jaw_to(jaw_width)
    elif g == 'rgt':
        m_rotmat = np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi * 1), b)
        m_rotmat = jaw_center_rotmat.dot(m_rotmat)
        gripper_m.rg_jaw_to(jaw_width)
    m_pos = np.array([-.06324, 0, -.19273]) + gripper.rotmat.dot(a)
    m_pos = jaw_center_pos + jaw_center_rotmat.dot(m_pos)
    gripper_m.fix_to(m_pos, m_rotmat)
    gripper_m.mg_open()
    gripper_m.gen_meshmodel(rgba=[0, 0, 1, .3]).attach_to(base)

base.run()


