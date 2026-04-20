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

# tool for planning finger grasps
tool_rec = put_object("tool_b_2", np.array([0, 0, 0]), np.eye(3), [.35, .35, .35, 1])
tool_cir = put_object("tool", np.array([0, 0, 0]), np.eye(3), [.35, .35, .35, 1])

# homo_matrix
T_rec_cir = make_homo(np.array([0, .065, -.09398]), rm.rotmat_from_axangle([0, 0, 1], -math.pi/2))
T_cir_rec = np.linalg.inv(T_rec_cir)

# =====================================planning for finger_1/2=====================================
rec_grasp_list = gpa.plan_grasps(lft_gripper, tool_rec,
                                 angle_between_contact_normals=math.radians(160),
                                 openning_direction='loc_x',
                                 max_samples=20, min_dist_between_sampled_contact_points=.005,
                                 contact_offset=.001)

pre_fin_1 = []
pre_fin_2 = []
pre_fin_3 = []
pre_fin_4 = []
for grasp_info in rec_grasp_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x_axis_direction = jaw_center_rotmat[:, 0]
    desired_direction = np.eye(3)[:, 1]
    cos_angle = (np.dot(x_axis_direction, desired_direction) /
                 (np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction)))

    # lft抓finger_1/2，放在finger_3/4，T_w_jaw = T_w_rec @ T_rec_jaw，其中T_w_rec = T_w_cir @ T_cir_rec
    if cos_angle == 1 or cos_angle == -1:
        T_w_rec_1 = make_homo(finger_1_pos, finger_1_rotmat) @ T_cir_rec
        T_w_jaw_1 = T_w_rec_1 @ make_homo(jaw_center_pos, jaw_center_rotmat)
        pre_fin_1.append([jaw_width, T_w_jaw_1[:3, 3], T_w_jaw_1[:3, :3]])

        T_w_rec_2 = make_homo(finger_2_pos, finger_2_rotmat) @ T_cir_rec
        T_w_jaw_2 = T_w_rec_2 @ make_homo(jaw_center_pos, jaw_center_rotmat)
        pre_fin_2.append([jaw_width, T_w_jaw_2[:3, 3], T_w_jaw_2[:3, :3]])

        T_w_rec_3 = make_homo(finger_3_pos, finger_3_rotmat) @ T_cir_rec
        T_w_jaw_3 = T_w_rec_3 @ make_homo(jaw_center_pos, jaw_center_rotmat)
        pre_fin_3.append([jaw_width, T_w_jaw_3[:3, 3], T_w_jaw_3[:3, :3]])

        T_w_rec_4 = make_homo(finger_4_pos, finger_4_rotmat) @ T_cir_rec
        T_w_jaw_4 = T_w_rec_4 @ make_homo(jaw_center_pos, jaw_center_rotmat)
        pre_fin_4.append([jaw_width, T_w_jaw_4[:3, 3], T_w_jaw_4[:3, :3]])

# =====================================planning for finger_3/4=====================================
cir_grasp_list = gpa.plan_grasps(lft_gripper, tool_cir,
                                 angle_between_contact_normals=math.radians(160),
                                 openning_direction='loc_x',
                                 max_samples=20, min_dist_between_sampled_contact_points=.005,
                                 contact_offset=.001)

fin_3 = []
fin_4 = []
for grasp_info in cir_grasp_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    x_axis_direction = jaw_center_rotmat[:, 0]
    desired_direction = np.eye(3)[:, 0]
    cos_angle = (np.dot(x_axis_direction, desired_direction) /
                 (np.linalg.norm(x_axis_direction) * np.linalg.norm(desired_direction)))

    # lft抓finger_3，T_w_jaw = T_w_cir @ T_cir_jaw
    if cos_angle == 1:
        T_w_jaw_3 = make_homo(finger_3_pos, finger_3_rotmat) @ make_homo(jaw_center_pos, jaw_center_rotmat)
        fin_3.append([jaw_width, T_w_jaw_3[:3, 3], T_w_jaw_3[:3, :3]])

    # rgt抓finger_4，T_w_jaw = T_w_cir @ T_cir_jaw
    if cos_angle == -1:
        T_w_jaw_4 = make_homo(finger_4_pos, finger_4_rotmat) @ make_homo(jaw_center_pos, jaw_center_rotmat)
        fin_4.append([jaw_width, T_w_jaw_4[:3, 3], T_w_jaw_4[:3, :3]])

gpa.write_pickle_file('finger_1', pre_fin_1, './pickle', 'pre_finger_1.pickle')
gpa.write_pickle_file('finger_2', pre_fin_2, './pickle', 'pre_finger_2.pickle')
gpa.write_pickle_file('finger_3', pre_fin_3, './pickle', 'pre_finger_3.pickle')
gpa.write_pickle_file('finger_4', pre_fin_4, './pickle', 'pre_finger_4.pickle')
gpa.write_pickle_file('finger_3', fin_3, './pickle', 'formal_finger_3.pickle')
gpa.write_pickle_file('finger_4', fin_4, './pickle', 'formal_finger_4.pickle')

base.run()

