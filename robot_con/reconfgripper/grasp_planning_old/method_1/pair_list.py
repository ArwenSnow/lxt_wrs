import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.robots.Franka_research3.Franka_research3 as Fr
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rg


# world
base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# robot
robot = Fr.Franka_research3()
robot.gen_meshmodel().attach_to(base)

# gripper
lft_gripper = rg.Reconfgripper().lft
rgt_gripper = rg.Reconfgripper().rgt
main_gripper = rg.Reconfgripper().body
gripper = rg.Reconfgripper()

# finger
finger_1 = cm.CollisionModel("../objects/finger_b_2.stl")
finger_1_pos = np.array([.3, .5, .015])
# finger_1_rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi/180*102.41)   # finger_a
finger_1_rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi/180*87.3)   # finger_b_1
finger_1.set_pos(finger_1_pos)
finger_1.set_rotmat(finger_1_rotmat)
gm.gen_frame(finger_1_pos, finger_1_rotmat).attach_to(base)
finger_1.set_rgba([.9, .75, .35, 1])
finger_1.attach_to(base)

finger_2 = cm.CollisionModel("../objects/finger_b_2.stl")
finger_2_pos = np.array([-.3, .5, .015])
finger_2_rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi/180*87.3)
finger_2.set_pos(finger_2_pos)
finger_2.set_rotmat(finger_2_rotmat)
gm.gen_frame(finger_2_pos, finger_2_rotmat).attach_to(base)
finger_2.set_rgba([.9, .75, .35, 1])
finger_2.attach_to(base)

# 对lft使用碰撞检测方案
lft_grasp_finger_list = gpa.load_pickle_file('finger_1', '../grasp_info/', 'lft_grasps.pickle')
objcm_list = [robot.base_stand.lnks[0]['collision_model']]
lft_valid_list = []
lft_tgt = []
lft_valid = 0

for lft_info in lft_grasp_finger_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = lft_info

    # finger_1 在世界坐标系下的位姿变化了，gl_jaw_center 随之变化
    new_jaw_center_pos = finger_1_rotmat @ jaw_center_pos + finger_1_pos
    new_jaw_center_rotmat = finger_1_rotmat @ jaw_center_rotmat
    lft_gripper.grip_at_with_jcpose(new_jaw_center_pos, new_jaw_center_rotmat, jaw_width)

    # 由 gl_lft 求 gl_gripper
    gl_lft_pos = lft_gripper.pos
    gl_lft_rotmat = lft_gripper.rotmat
    g_pos = gl_lft_rotmat @ np.array([-.053, .018, -.132]) + gl_lft_pos
    g_rotmat = gl_lft_rotmat
    gripper.fix_to(g_pos, g_rotmat)

    # 碰撞检测，保留不撞桌子的方案
    if not gripper.is_mesh_collided(objcm_list=objcm_list):
        lft_valid += 1
        lft_valid_list.append(lft_info)
        lft_tgt.append([g_pos, g_rotmat, lft_info[0]])

        gripper.lg_jaw_to(jaw_width)
        conf = robot.tracik(tgt_pos=g_pos, tgt_rotmat=g_rotmat)
        robot.fk("arm", conf)
        robot.hnd.lg_jaw_to(jaw_width)
        # robot.gen_meshmodel().attach_to(base)

# 对rgt使用碰撞检测方案
rgt_grasp_finger_list = gpa.load_pickle_file('finger_2', '../grasp_info/', 'rgt_grasps.pickle')
rgt_valid_list = []
rgt_tgt = []
rgt_valid = 0

for rgt_info in rgt_grasp_finger_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = rgt_info

    # finger_2 在世界坐标系下的位姿变化了，gl_jaw_center 随之变化
    new_jaw_center_pos = finger_2_rotmat @ jaw_center_pos + finger_2_pos
    new_jaw_center_rotmat = finger_2_rotmat @ jaw_center_rotmat
    rgt_gripper.grip_at_with_jcpose(new_jaw_center_pos, new_jaw_center_rotmat, jaw_width)

    # 由 gl_rgt 求 gl_gripper
    gl_rgt_pos = rgt_gripper.pos
    gl_rgt_rotmat = rgt_gripper.rotmat
    g_pos = gl_rgt_rotmat @ np.array([.053, -.018, -.132]) + gl_rgt_pos
    g_rotmat = gl_rgt_rotmat
    gripper.fix_to(g_pos, g_rotmat)

    # 碰撞检测，保留不撞桌子的方案
    if not gripper.is_mesh_collided(objcm_list=objcm_list):
        rgt_valid += 1
        rgt_valid_list.append(rgt_info)
        rgt_tgt.append([g_pos, g_rotmat, rgt_info[0]])

        gripper.rg_jaw_to(jaw_width)
        conf = robot.tracik(tgt_pos=g_pos, tgt_rotmat=g_rotmat)
        robot.fk("arm", conf)
        robot.hnd.rg_jaw_to(jaw_width)
        # robot.gen_meshmodel().attach_to(base)

print("lft有效抓取方案数量：", lft_valid)
print("rgt有效抓取方案数量：", rgt_valid)

# 选取lft和rgt相对应的抓取方案
pair = 0
pair_list = []
for lft_idx, i in enumerate(lft_valid_list):
    _, lft_pos, lft_rot, _, _ = i
    lft_x = lft_rot[:, 0]
    lft_y = lft_rot[:, 1]
    for rgt_idx, j in enumerate(rgt_valid_list):
        _, rgt_pos, rgt_rot, _, _ = j
        rgt_x = rgt_rot[:, 0]
        rgt_y = rgt_rot[:, 1]
        x_cos = (np.dot(lft_x, rgt_x) / (np.linalg.norm(lft_x) * np.linalg.norm(rgt_x)))
        y_cos = (np.dot(lft_y, rgt_y) / (np.linalg.norm(lft_y) * np.linalg.norm(rgt_y)))
        if np.allclose(lft_pos, rgt_pos) and abs(x_cos + 1) < 1e-6 and abs(y_cos + 1) < 1e-6:
            pair += 1
            pair_list.append([lft_tgt[lft_idx], rgt_tgt[rgt_idx]])
print("左右对称抓取方案的数量", pair)
gpa.write_pickle_file('finger', pair_list, './', 'lft_rgt_grasps.pickle')

base.run()


