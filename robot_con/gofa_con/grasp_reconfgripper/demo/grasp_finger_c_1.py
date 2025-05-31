import copy
import math
from time import sleep
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa

import robot_sim.robots.gofa5.gofa5 as gf5
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as reconf
import robot_sim.end_effectors.gripper.xc330gripper1.xc330gripper1 as xcgf
import motion.probabilistic.rrt_connect as rrtc

import robot_con.gofa_con.gofa_con as gofa_con
import robot_con.reconfgripper.maingripper.maingripper as dh
import robot_con.reconfgripper.xc330gripper.xc330gripper as xc


def go_init():
    init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_jnts = rbt_s.get_jnt_values("arm")

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
    xc_r = xc.Xc330Gripper(xc_s, 'COM3', 57600, real=True)
    dh_r = dh.MainGripper()
    go_init()
    print("机器人初始化完成")
    xc_r.init_both_gripper()
    sleep(15)
    print("小夹爪初始化完成")
    dh_r.init_gripper()
    sleep(4)
    print("大夹爪初始化完成")

    # 1.start → app_finger1
    start_jnts = np.array([0, 0, 0, 0, 0, 0])
    app1_jnts = np.array([-0.24678956, 0.23666665, 0.62570054, -1.61896741, -0.12391838, -0.22060962])
    path_app_1 = rrtc_s.plan(component_name="arm",
                             start_conf=start_jnts,
                             goal_conf=app1_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_app_1)

    # 2.app_finger1 → gri_finger1
    gri1_jnts = np.array([-0.24923302, 0.59445914, 0.48415433, -1.61844382, -0.0715585, -0.22410028])
    path_gri_1 = rrtc_s.plan(component_name="arm",
                             start_conf=app1_jnts,
                             goal_conf=gri1_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_gri_1)
    xc_r.lg_grasp_with_force(-15)
    sleep(10)

    # 3.gri_finger1 → app_finger1
    rbt_r.move_jntspace_path(path_gri_1[::-1])

    # 4.app_finger1 → app_finger2
    app2_jnts = np.array([-0.36267942, 0.22968533, 0.62412974, -1.63816604, -0.07609636, -0.16894787])
    path_app_2 = rrtc_s.plan(component_name="arm",
                             start_conf=app1_jnts,
                             goal_conf=app2_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_app_2)

    # 5.app_finger2 → gri_finger2
    gri2_jnts = np.array([-0.36634461, 0.55745816, 0.54733525, -1.55177224, -0.12077678, -0.12479104])
    path_gri_2 = rrtc_s.plan(component_name="arm",
                             start_conf=app2_jnts,
                             goal_conf=gri2_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_gri_2)
    xc_r.rg_grasp_with_force(-15)
    sleep(10)

    # 6.gri_finger2 → app_finger2
    rbt_r.move_jntspace_path(path_gri_2[::-1])

    # 7.app_finger2 → app_object
    app3_jnts = np.array([-0.03211406, 0.41434116, 0.4420919, -1.62141088, -0.09599311, -0.15428711])
    path_app_3 = rrtc_s.plan(component_name="arm",
                             start_conf=app2_jnts,
                             goal_conf=app3_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_app_3)

    # 8.app_object → gri_object
    gri3_jnts = np.array([0.00174533, 0.62954026, 0.39741147, -1.69611097, -0.03333579, 0.02111848])
    path_gri_3 = rrtc_s.plan(component_name="arm",
                             start_conf=app3_jnts,
                             goal_conf=gri3_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_gri_3)
    dh_r.jaw_to(0)
    sleep(2)

    # 9.gri_object → app_object
    rbt_r.move_jntspace_path(path_gri_3[::-1])

    base.run()



