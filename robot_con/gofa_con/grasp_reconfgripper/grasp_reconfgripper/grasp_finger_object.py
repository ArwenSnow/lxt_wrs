import time
import os
import pickle

import numpy as np
import visualization.panda.world as wd
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.reconfgrippper.reconfgripper as hnd
import robot_sim.robots.gofa5.gofa5 as gf5
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import robot_con.gofa_con.grasp_reconfgripper.animation as genani
import robot_con.gofa_con.grasp_reconfgripper.grasp_finger as grafin

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])

    # robot
    rbt_s = gf5.GOFA5()

    # finger_1
    obj_path = f"../../../0000_examples/objects/finger_a.stl"
    finger_1 = cm.CollisionModel(obj_path, expand_radius=-0.00)
    finger_1_pos = np.array([0.6, 0, 0.025])
    finger_1.set_pos(finger_1_pos)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(np.radians(90)), -np.sin(np.radians(90))],
                    [0, np.sin(np.radians(90)), np.cos(np.radians(90))]])
    R_z = np.array([[np.cos(np.radians(90)), -np.sin(np.radians(90)), 0],
                    [np.sin(np.radians(90)), np.cos(np.radians(90)), 0],
                    [0, 0, 1]])
    finger_1_rotmat = R_z.dot(R_x)
    finger_1.set_rotmat(finger_1_rotmat)

    # finger_2
    finger_2 = cm.CollisionModel(obj_path, expand_radius=0.00)
    finger_2_pos = np.array([0.7, .2, 0.015])
    finger_2.set_pos(finger_2_pos)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(np.radians(270)), -np.sin(np.radians(270))],
                    [0, np.sin(np.radians(270)), np.cos(np.radians(270))]])
    R_z = np.array([[np.cos(np.radians(315)), -np.sin(np.radians(315)), 0],
                    [np.sin(np.radians(315)), np.cos(np.radians(315)), 0],
                    [0, 0, 1]])
    finger_2_rotmat = R_z.dot(R_x)
    finger_2.set_rotmat(finger_2_rotmat)

    full_path = gpa.load_pickle_file('robot_path', './path_list/full_path/', 'robot_path.pickle')
    mg_jawwidth_list = gpa.load_pickle_file('mg_path', './path_list/full_path/', 'mg_path.pickle')
    lft_jawwidth_list = gpa.load_pickle_file('lft_path', './path_list/full_path/', 'lft_path.pickle')
    rgt_jawwidth_list = gpa.load_pickle_file('rgt_path', './path_list/full_path/', 'rgt_path.pickle')
    objpose_list_1 = gpa.load_pickle_file('finger_1_path', './path_list/full_path/', 'finger_1_path.pickle')
    objpose_list_2 = gpa.load_pickle_file('finger_2_path', './path_list/full_path/', 'finger_2_path.pickle')
    robot_attached_list = []
    object_attached_list = []
    counter = [0]

    taskMgr.doMethodLater(0.1, genani.update, "update",
                          extraArgs=[rbt_s,
                                     finger_1,
                                     finger_2,
                                     full_path,
                                     mg_jawwidth_list,
                                     lft_jawwidth_list,
                                     rgt_jawwidth_list,
                                     objpose_list_1,
                                     objpose_list_2,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    time.sleep(2)

    base.run()


