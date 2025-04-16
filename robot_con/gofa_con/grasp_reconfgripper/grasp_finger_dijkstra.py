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


def calculate_pos_similarity(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def calculate_rotmat_similarity(rotmat1, rotmat2):
    dot_product = np.trace(np.dot(rotmat1.T, rotmat2))
    angle = np.arccos((dot_product - 1) / 2)
    return angle


def cost(grasp1, grasp2, w_pos=1.0, w_rot=1.0):
    _, pos1, rotmat1, _, _ = grasp1
    _, pos2, rotmat2, _, _ = grasp2
    pos_diff = calculate_pos_similarity(pos1, pos2)
    rot_diff = calculate_rotmat_similarity(rotmat1, rotmat2)
    edge_weight = w_pos * pos_diff + w_rot * rot_diff
    return edge_weight


if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    graph = {}


    base.run()

