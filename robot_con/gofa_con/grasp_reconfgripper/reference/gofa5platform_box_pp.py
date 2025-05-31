import copy
import math
import time

import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa

import robot_sim.robots.gofa5.gofa5 as gf5
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as reconf
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xcgf
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xc
import motion.probabilistic.rrt_connect as rrtc

import basis.robot_math as rm
import robot_con.gofa_con.gofa_con as gofa_con
import robot_con.reconfgripper.maingripper.maingripper as dh
import robot_con.reconfgripper.xc330gripper.xc330gripper as xc


def go_init():
    init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_jnts = rbt_s.get_jnt_values("arm")

    # 令当前位置回到初始位置
    path = rrtc_s.plan(component_name="arm",
                       start_conf=current_jnts,
                       goal_conf=init_jnts,
                       ext_dist=0.05,
                       max_time=300)
    rbt_r.move_jntspace_path(path)


if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    rbt_s = gf5.GOFA5()
    dh_s = reconf.reconfgripper()
    xc_s = xcgf.xc330gripper()
    rrtc_s = rrtc.RRTConnect(rbt_s)

    rbt_r = gofa_con.GoFaArmController()
    # dh_r = dh.MainGripper()
    xc_r = xc.Xc330Gripper(xc_s, 'COM3', 57600, real=True)

    go_init()
    print("初始化完成")

    start_jnts = np.array([0, 0, 0, 0, 0, 0])
    goal_jnts = np.array([-0.13875368, -0.30927234, 1.08524573, -0.0439823, 0.1144936, 0.0015708])

    path_1 = rrtc_s.plan(component_name="arm",
                         start_conf=start_jnts,
                         goal_conf=goal_jnts,
                         ext_dist=0.05,
                         max_time=300)

    # rbt_r.move_jntspace_path(path_1)      # start                →  approach to finger_1
    # time.sleep(2)
    # xc_r.init_lg()                        # lg_init
    # time.sleep(10)

    # rbt_r.move_jntspace_path(path_1)      # approach to finger_1 →  get finger_1
    # time.sleep(2)
    # xc_r.lg_grasp_with_force(-45)  # lg jaw to 0

    # rbt_r.move_jntspace_path(path_1)      # get finger_1         →  approach to finger_1
    # time.sleep(2)
    #
    # rbt_r.move_jntspace_path(path_1)      # approach to finger_1 →  approach to finger_2
    # time.sleep(2)
    # xc_r.init_rg()
    # time.sleep(15)
    # xc_r.rg_grasp_with_force(-45)
    #
    # rbt_r.move_jntspace_path(path_1)      # approach to finger_2 →  get finger_2
    # time.sleep(2)

    base.run()