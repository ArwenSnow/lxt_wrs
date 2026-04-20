import time

import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import motion.probabilistic.rrt_connect as rrtc
import robot_sim.robots.Franka_research3.Franka_research3 as Fr
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as rg


def update_robot_joints(jnt_values):
    """
    根据关节角度更新机器人状态
    """
    global current_robot_mesh
    try:
        if current_robot_mesh is not None:
            current_robot_mesh.detach()
        robot_s.fk(jnt_values=jnt_values)
        new_mesh = robot_s.gen_meshmodel()
        new_mesh.attach_to(base)
        current_robot_mesh = new_mesh
        return True
    except Exception as e:
        print(f"更新机器人时出错: {e}")
        return False


# world
base = wd.World(cam_pos=[-2, 4, 1.5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

# table
table = cm.CollisionModel("../objects/table.stl")
table.set_pos(np.array([0, 0, -1]))
table.attach_to(base)

# robot
robot_s = Fr.Franka_research3()
current_robot_mesh = robot_s.gen_meshmodel()
current_robot_mesh.attach_to(base)

# rrtc planner
rrtc_planner = rrtc.RRTConnect(robot_s)

# gripper
lft_gripper = rg.Reconfgripper().lft
rgt_gripper = rg.Reconfgripper().rgt
main_gripper = rg.Reconfgripper().body
gripper = rg.Reconfgripper()

# finger
finger_1 = cm.CollisionModel("../objects/finger_b_2.stl")
finger_1_pos = np.array([.3, .5, .015])
finger_1_rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi/180*87.3)
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

path = []
path_1 = []
path_2 = []
path_3 = []
path_4 = []
pair_list = gpa.load_pickle_file('finger', './', 'lft_rgt_grasps.pickle')
for finger_grasp in pair_list:
    first_grasp, second_grasp = finger_grasp
    first_pos, first_rotmat, first_jawwidth = first_grasp
    second_pos, second_rotmat, second_jawwidth = second_grasp

    # 抓第一只手指，先到上空10cm。
    first_pos_prepare = first_pos.copy()
    first_pos_prepare[2] += 0.1
    first_conf_prepare = robot_s.tracik(tgt_pos=first_pos_prepare, tgt_rotmat=first_rotmat)
    path_1 = rrtc_planner.plan(component_name="arm",
                               start_conf=np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, math.pi/4*3]),
                               goal_conf=first_conf_prepare,
                               obstacle_list=[table],
                               otherrobot_list=[],
                               ext_dist=0.05)
    first_conf = robot_s.tracik(tgt_pos=first_pos, tgt_rotmat=first_rotmat)
    path_2 = rrtc_planner.plan(component_name="arm",
                               start_conf=first_conf_prepare,
                               goal_conf=first_conf,
                               obstacle_list=[table],
                               otherrobot_list=[],
                               ext_dist=0.05)
    robot_s.fk("arm", first_conf)
    robot_s.hnd.lg_jaw_to(first_jawwidth)
    # robot_s.gen_meshmodel().attach_to(base)

    # 抓第二只手指
    second_pos_prepare = second_pos.copy()
    second_pos_prepare[2] += 0.1
    second_conf_prepare = robot_s.tracik(tgt_pos=second_pos_prepare, tgt_rotmat=second_rotmat)
    path_3 = rrtc_planner.plan(component_name="arm",
                               start_conf=first_conf_prepare,
                               goal_conf=second_conf_prepare,
                               obstacle_list=[table],
                               otherrobot_list=[],
                               ext_dist=0.05)
    second_conf = robot_s.tracik(tgt_pos=second_pos, tgt_rotmat=first_rotmat)
    path_4 = rrtc_planner.plan(component_name="arm",
                               start_conf=second_conf_prepare,
                               goal_conf=second_conf,
                               obstacle_list=[table],
                               otherrobot_list=[],
                               ext_dist=0.05)
    robot_s.fk("arm", second_conf)
    robot_s.hnd.rg_jaw_to(second_jawwidth)
    # robot_s.gen_meshmodel().attach_to(base)
    break


count = 1
reversible_counter = 1
path.extend(path_1)
path.extend(path_2)
path.extend(reversed(path_2))
path.extend(path_3)
path.extend(path_4)
path.extend(reversed(path_4))


def update_task(task):
    global count
    jnt_values = path[count]
    if jnt_values is not None:
        update_robot_joints(jnt_values)
    count += 1
    if count >= len(path):
        time.sleep(1)
        count = 0
    return task.again


try:
    taskMgr.doMethodLater(0.1, update_task, "update")
    base.run()
except KeyboardInterrupt:
    print("\n服务器被用户中断")

base.run()

