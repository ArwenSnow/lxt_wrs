import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rg


# ======================================function======================================
def put_object(name, obj_pos, obj_rot, color):
    obj = cm.CollisionModel(f"../objects/{name}.stl")
    obj.set_pos(obj_pos)
    obj.set_rotmat(obj_rot)
    obj.set_rgba(color)
    return obj


def make_homo(pos, rotmat):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = pos
    return homo


# ====================================load_pickle====================================
fin_1 = gpa.load_pickle_file('finger_1', './', 'finger_1.pickle')
fin_2 = gpa.load_pickle_file('finger_2', './', 'finger_2.pickle')
fin_3 = gpa.load_pickle_file('finger_3', './', 'finger_3.pickle')
fin_4 = gpa.load_pickle_file('finger_4', './', 'finger_4.pickle')

# =======================================object=======================================
# world
base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# table
table = put_object("table", np.array([0, 0, 0]), np.eye(3), [.35, .35, .35, 1])
table.attach_to(base)

# gripper
lft_gripper = rg.Reconfgripper().lft
rgt_gripper = rg.Reconfgripper().rgt
gripper = rg.Reconfgripper()

# target
tar_pos = np.array([-.4, .85, 0])
tar_rot = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2)
target = put_object("box", tar_pos, tar_rot, [.9, .75, .35, 1])
target.attach_to(base)

# finger
finger_1_pos = np.array([-.3, .5, .015])
finger_1_rotmat = (rm.rotmat_from_axangle([0, 0, 1], math.pi / 2 * 1.2)
                   @ rm.rotmat_from_axangle([1, 0, 0], math.pi / 180 * 87.3))
finger_2_pos = np.array([.3, .5, .015])
finger_2_rotmat = (rm.rotmat_from_axangle([1, 0, 0], -math.pi / 180 * 87.3)
                   @ rm.rotmat_from_axangle([0, 1, 0], math.pi / 2 * 0.5))
finger_3_pos = np.array([-.0265, .75, .015])
finger_3_rotmat = (rm.rotmat_from_axangle([1, 0, 0], -math.pi / 180 * 87.3)
                   @ rm.rotmat_from_axangle([0, 0, 1], math.pi))
finger_4_pos = np.array([.0265, .75, .015])
finger_4_rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 180 * 87.3)

finger_1 = put_object("finger_b_2", finger_1_pos, finger_1_rotmat, [.9, .75, .35, 1])
finger_2 = put_object("finger_b_2", finger_2_pos, finger_2_rotmat, [.9, .75, .35, 1])
finger_3 = put_object("finger_b_2", finger_3_pos, finger_3_rotmat, [.9, .75, .35, 1])
finger_4 = put_object("finger_b_2", finger_4_pos, finger_4_rotmat, [.9, .75, .35, 1])
finger_1.attach_to(base)
finger_2.attach_to(base)
finger_3.attach_to(base)
finger_4.attach_to(base)

# homo_matrix
T_lft_gri = make_homo(np.array([-.053, .018, -.132]), np.eye(3))
T_gri_lft = np.linalg.inv(T_lft_gri)
T_rgt_gri = make_homo(np.array([.053, -.018, -.132]), np.eye(3))
T_gri_rgt = np.linalg.inv(T_rgt_gri)

# ===================collision detection for finger_1/2/3/4======================================
finger_1_list = []
finger_2_list = []
finger_3_list = []
finger_4_list = []

# 存储抓finger_1不碰撞的gri_pose，T_w_gri = T_w_lft @ T_lft_gri
for fin_1_info in fin_1:
    jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat = fin_1_info
    lft_gripper.grip_at_with_jcpose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)
    T_w_gri = make_homo(lft_gripper.pos, lft_gripper.rotmat) @ T_lft_gri
    gripper.fix_to(T_w_gri[:3, 3], T_w_gri[:3, :3])
    gripper.lg_jaw_to(jaw_width)
    if not gripper.is_mesh_collided(objcm_list=[table, finger_1]):
        finger_1_list.append([jaw_width, T_w_gri[:3, 3], T_w_gri[:3, :3]])
        gripper.gen_meshmodel().attach_to(base)

# 存储抓finger_2不碰撞的gri_pose，T_w_gri = T_w_lft @ T_lft_gri
for fin_2_info in fin_2:
    jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat = fin_2_info
    lft_gripper.grip_at_with_jcpose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)
    T_w_gri = make_homo(lft_gripper.pos, lft_gripper.rotmat) @ T_lft_gri
    gripper.fix_to(T_w_gri[:3, 3], T_w_gri[:3, :3])
    gripper.lg_jaw_to(jaw_width)
    if not gripper.is_mesh_collided(objcm_list=[table, finger_2]):
        finger_2_list.append([jaw_width, T_w_gri[:3, 3], T_w_gri[:3, :3]])
        gripper.gen_meshmodel().attach_to(base)

# 存储抓finger_3不碰撞的gri_pose，T_w_gri = T_w_lft @ T_lft_gri
for fin_3_info in fin_3:
    jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat = fin_3_info
    lft_gripper.grip_at_with_jcpose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)
    T_w_gri = make_homo(lft_gripper.pos, lft_gripper.rotmat) @ T_lft_gri
    gripper.fix_to(T_w_gri[:3, 3], T_w_gri[:3, :3])
    gripper.lg_jaw_to(jaw_width)
    gripper.rg_jaw_to(jaw_width)
    if not gripper.is_mesh_collided(objcm_list=[table, finger_3, finger_4]):
        finger_3_list.append([jaw_width, T_w_gri[:3, 3], T_w_gri[:3, :3]])
        gripper.gen_meshmodel().attach_to(base)

# 存储抓finger_4不碰撞的gri_pose，T_w_gri = T_w_lft @ T_lft_gri
for fin_4_info in fin_4:
    jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat = fin_4_info
    rgt_gripper.grip_at_with_jcpose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)
    T_w_gri = make_homo(rgt_gripper.pos, rgt_gripper.rotmat) @ T_rgt_gri
    gripper.fix_to(T_w_gri[:3, 3], T_w_gri[:3, :3])
    gripper.lg_jaw_to(jaw_width)
    gripper.rg_jaw_to(jaw_width)
    if not gripper.is_mesh_collided(objcm_list=[table, finger_3, finger_4]):
        finger_4_list.append([jaw_width, T_w_gri[:3, 3], T_w_gri[:3, :3]])
        gripper.gen_meshmodel().attach_to(base)

gpa.write_pickle_file('finger_1', finger_1_list, './', 'finger_1_cd.pickle')
gpa.write_pickle_file('finger_2', finger_2_list, './', 'finger_2_cd.pickle')
gpa.write_pickle_file('finger_3', finger_3_list, './', 'finger_3_cd.pickle')
gpa.write_pickle_file('finger_4', finger_4_list, './', 'finger_4_cd.pickle')

base.run()



