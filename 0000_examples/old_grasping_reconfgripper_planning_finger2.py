##reconfgripper grasps two fingers
import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper_old.reconfgripper as rf

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# object
lft_finger = cm.CollisionModel("objects/finger.stl")
lft_finger.set_rgba([.35, .35, .35, 1])
lft_finger.attach_to(base)

rgt_finger = cm.CollisionModel("objects/finger.stl")
rgt_finger.set_rotmat(np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi * 1), lft_finger.get_rotmat()))
rgt_finger.set_rgba([.35, .35, .35, 1])

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
contact_offset = .0025
jaw_width = .008 + contact_offset*2
angle_increment = math.pi*45/180
grasp_info_list = []


for i in range(8):
    lft_jaw_center_rotmat = np.dot(rm.rotmat_from_axangle([1, 0, 0], i*angle_increment ),c)
    gripper_l.grip_at_with_jcpose(lft_jaw_center_pos, lft_jaw_center_rotmat, jaw_width)
    # gripper_l.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)

    m_rotmat = lft_jaw_center_rotmat
    # gripper_m.lg_jaw_to(jaw_width)
    # gripper_m.rg_jaw_to(jaw_width)
    gripper_m.mg_jaw_to(.076)
    e = gripper_m.get_jawwidth()/2
    m_pos = np.array([-.02524-e, 0, -.19273]) + gripper_l.rotmat.dot(a)
    m_pos = lft_jaw_center_pos + lft_jaw_center_rotmat.dot(m_pos)


    d = gripper_m.get_jawwidth()
    rgt_finger.set_pos(lft_finger.get_pos()+np.array([-(d+.05048), 0, 0]))
    rgt_finger.attach_to(base)

    rgt_jaw_center_pos = lft_jaw_center_pos + np.array([-(d+.05048), 0, 0])
    rgt_jaw_center_rotmat = np.dot(rm.rotmat_from_axangle([1, 0, 0], -i * angle_increment), c)
    rgt_jaw_center_rotmat = np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi ),rgt_jaw_center_rotmat)
    gripper_r.grip_at_with_jcpose(rgt_jaw_center_pos, rgt_jaw_center_rotmat, jaw_width)
    # gripper_r.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(base)

    grasp_info_list.append([
        lft_jaw_center_pos,
        lft_jaw_center_rotmat,
        rgt_jaw_center_pos,
        rgt_jaw_center_rotmat,
        m_pos,
        m_rotmat,
        jaw_width
    ])
grasp_info_list.pop(4)

# for grasp_info in grasp_info_list:
#     lft_pos, lft_rotmat, rgt_pos, rgt_rotmat, m_pos, m_rotmat, jaw_width = grasp_info
#     gripper_l.grip_at_with_jcpose(lft_pos, lft_rotmat, jaw_width)
#     gripper_l.gen_meshmodel().attach_to(base)
#     gripper_r.grip_at_with_jcpose(rgt_pos, rgt_rotmat, jaw_width)
#     gripper_r.gen_meshmodel().attach_to(base)
#     gripper_m.fix_to(m_pos, m_rotmat)
#     gripper_m.gen_meshmodel().attach_to(base)

grasp_info = grasp_info_list[0]
lft_pos, lft_rotmat, rgt_pos, rgt_rotmat, m_pos, m_rotmat, jaw_width = grasp_info
gripper_l.grip_at_with_jcpose(lft_pos, lft_rotmat, jaw_width)
gripper_l.gen_meshmodel().attach_to(base)
gripper_r.grip_at_with_jcpose(rgt_pos, rgt_rotmat, jaw_width)
gripper_r.gen_meshmodel().attach_to(base)
gripper_m.fix_to(m_pos, m_rotmat)
gripper_m.gen_meshmodel().attach_to(base)

base.run()


