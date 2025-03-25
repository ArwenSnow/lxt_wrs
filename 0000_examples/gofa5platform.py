import copy
import math
import time

import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
# import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.end_effectors.gripper.ag145.ag145 as dh
import robot_sim.robots.gofa5.GOFA5 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import robot_con.gofa_con.gofa_con as gofa_con

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
    rbt_r = gofa_con.GoFaArmController()
    rrtc_s = rrtc.RRTConnect(rbt_s)

    # start_conf = np.array([0, 0, 0, 0, 0, 0])
    # go_init()  # 令机械臂从当前位置回到机械臂初始位置
    # print("hi")
    # base.run()
    #
    # goal_jnt_values = [-0.09778316, -0.49875734, -0.6381371, 1.02269445, -1.07121682, -2.65028371]
    # path_1 = rrtc_s.plan(component_name="arm",
    #                          start_conf=start_conf,
    #                          goal_conf=goal_jnt_values,
    #                          ext_dist=0.05,
    #                          max_time=300)
    # rbt_r.move_jntspace_path(path_1)
    #
    # base.run()


    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    rbt_s = gf5.GOFA5()
    rbt_r = gofa_con.GoFaArmController()
    current_jnts = rbt_r.get_jnt_values()
    print(current_jnts)
    # start_conf = np.array([0.0439823, -0.53023103, 1.05243354, 0.0143117, 1.55351757, 1.57079633])

    # [0.12269665 - 0.00436332  0.64385196 - 0.1272345   0.99117248 - 0.00191986]

    start_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rbt_s.fk("arm", start_conf)
    # rbt_s.fk("arm",np.asarray(rbt_r.get_jnt_values()))
    rbt_s.gen_meshmodel().attach_to(base)
    # base.run()
    rbt_r.move_j(start_conf)
    # base.run()
    start_pos, start_rot = rbt_s.get_gl_tcp("arm")
    goal_pos = start_pos + [0.05, 0, 0]
    goal_rot = start_rot
    goal_jnts = rbt_s.ik("arm", goal_pos, goal_rot)

    rbt_r.move_j(goal_jnts)
    base.run()