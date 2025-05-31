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
    go_init()
    print("机器人初始化完成")

    # 1.start → app_finger1
    start_jnts = np.array([0, 0, 0, 0, 0, 0])
    app1_jnts = np.array([-0.24678956, 0.23666665, 0.62570054, -1.61896741, -0.12391838, -0.22060962])
    # app1_jnts = np.array([-0.24923302, 0.59445914, 0.48415433, -1.61844382, -0.0715585, -0.22410028])
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

    # 3.gri_finger1 → app_finger1
    rbt_r.move_jntspace_path(path_gri_1[::-1])

    # 4.app_finger1 → app_finger2
    app2_jnts = np.array([0.09686577, 0.22549654, 0.625526, -1.61914195, -0.12322025, -0.17243853])
    path_app_2 = rrtc_s.plan(component_name="arm",
                             start_conf=app1_jnts,
                             goal_conf=app2_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_app_2)

    # 5.app_finger2 → gri_finger2
    gri2_jnts = np.array([0.15271631, 0.57299159, 0.51173054, -1.67569061, -0.10768681, -0.07958701])
    path_gri_2 = rrtc_s.plan(component_name="arm",
                             start_conf=app2_jnts,
                             goal_conf=gri2_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_gri_2)

    # 6.gri_finger2 → app_finger2
    rbt_r.move_jntspace_path(path_gri_2[::-1])

    # 7.app_finger2 → app_object
    app3_jnts = np.array([-0.51661746, 0.34924038, 0.31869712, 0.11030481, 0.82309728, -1.66312424])
    path_app_3 = rrtc_s.plan(component_name="arm",
                             start_conf=app2_jnts,
                             goal_conf=app3_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_app_3)

    # 8.app_object → gri_object
    gri3_jnts = np.array([-0.50579642, 0.65380034, -0.10821041, -0.00645772, 1.27356676, -1.61547676])
    path_gri_3 = rrtc_s.plan(component_name="arm",
                             start_conf=app3_jnts,
                             goal_conf=gri3_jnts,
                             ext_dist=0.05,
                             max_time=300)
    rbt_r.move_jntspace_path(path_gri_3)

    # 9.gri_object → app_object
    rbt_r.move_jntspace_path(path_gri_3[::-1])

    base.run()



