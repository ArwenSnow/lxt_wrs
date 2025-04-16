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
from collections import defaultdict
import heapq


def calculate_pos_similarity(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def calculate_rotmat_similarity(rotmat1, rotmat2):
    dot_product = np.trace(np.dot(rotmat1.T, rotmat2))
    angle = np.arccos((dot_product - 1) / 2)
    return angle


def cost(grasp1, grasp2, w_pos=.2, w_rot=.8):
    l_pos, l_rotmat, pos_1, rotmat_1 = grasp1
    r_pos, r_rotmat, pos_2, rotmat_2 = grasp2

    f_pos_1 = pos_1 + np.dot(rotmat_1, l_pos)
    f_pos_2 = pos_2 + np.dot(rotmat_2, r_pos)
    pos_diff = calculate_pos_similarity(f_pos_1, f_pos_2)

    f_rotmat_1 = rotmat_1.dot(l_rotmat)
    f_rotmat_2 = rotmat_2.dot(r_rotmat)
    rot_diff = calculate_rotmat_similarity(f_rotmat_1, f_rotmat_2)

    edge_weight = w_pos * pos_diff + w_rot * rot_diff
    return edge_weight


# def dijkstra(graph, start, goal):
#     queue = []
#     heapq.heappush(queue, (0, start))  # (cost, node)
#     came_from = {start: None}
#     cost_so_far = {start: 0}
#
#     while queue:
#         current_cost, current_node = heapq.heappop(queue)
#
#         if current_node == goal:
#             break
#
#         for neighbor in graph[current_node]:
#             new_cost = current_cost + graph[current_node][neighbor]
#             if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
#                 cost_so_far[neighbor] = new_cost
#                 heapq.heappush(queue, (new_cost, neighbor))
#                 came_from[neighbor] = current_node
#
#     # reconstruct path
#     if goal not in came_from:
#         return None  # no path
#
#     path = []
#     node = goal
#     while node:
#         path.append(node)
#         node = came_from[node]
#     path.reverse()
#     return path

def dijkstra(graph, start, goal, edge_info=None):
    queue = []
    heapq.heappush(queue, (0, start))  # (cost, node)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while queue:
        current_cost, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor in graph[current_node]:
            new_cost = current_cost + graph[current_node][neighbor]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
                came_from[neighbor] = current_node

    # reconstruct path
    if goal not in came_from:
        return None, None  # no path

    path = []
    node = goal
    while node:
        path.append(node)
        node = came_from[node]
    path.reverse()

    # reconstruct edge-based pose info
    center_pairs = []
    if edge_info is not None:
        for i in range(len(path) - 1):
            key = (path[i], path[i+1])
            if key in edge_info:
                center_pairs.append(edge_info[key])

    return path, center_pairs


if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    obj_path = f"../../../0000_examples/objects/finger_a.stl"

    # set finger_1 pose
    finger_1 = cm.CollisionModel(obj_path, expand_radius=-0.001)
    finger_1_pos = np.array([0.6, .1, 0.015])
    finger_1.set_pos(finger_1_pos)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(np.radians(90)), -np.sin(np.radians(90))],
                    [0, np.sin(np.radians(90)), np.cos(np.radians(90))]])
    R_z = np.array([[np.cos(np.radians(90)), -np.sin(np.radians(90)), 0],
                    [np.sin(np.radians(90)), np.cos(np.radians(90)), 0],
                    [0, 0, 1]])
    finger_1_rotmat = R_z.dot(R_x)
    finger_1.set_rotmat(finger_1_rotmat)

    # set finger_2 pose
    finger_2 = cm.CollisionModel(obj_path, expand_radius=0.00)
    finger_2_pos = np.array([0.8, .2, 0.015])
    finger_2.set_pos(finger_2_pos)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(np.radians(270)), -np.sin(np.radians(270))],
                    [0, np.sin(np.radians(270)), np.cos(np.radians(270))]])
    R_z = np.array([[np.cos(np.radians(315)), -np.sin(np.radians(315)), 0],
                    [np.sin(np.radians(315)), np.cos(np.radians(315)), 0],
                    [0, 0, 1]])
    finger_2_rotmat = R_z.dot(R_x)
    finger_2.set_rotmat(finger_2_rotmat)

    # rbt_s and gripper_s
    rbt_s = gf5.GOFA5()
    gripper_s = hnd.reconfgripper(pos=rbt_s.arm.jnts[-1]['gl_posq'],
                                  rotmat=rbt_s.arm.jnts[-1]['gl_rotmatq'],
                                  name='hnd', enable_cc=False)
    grasp_info_list = gpa.load_pickle_file('finger', '../../../0000_examples/', 'reconfgripper_finger_grasps.pickle')

    finger_1_homo = rm.homomat_from_posrot(finger_1_pos, finger_1_rotmat)
    target_conf_1_list = []
    approach_conf_1_list = []
    lftjcenter = []
    lli = []
    lfor_grasp_object = []
    ldijkstra = []

    finger_2_homo = rm.homomat_from_posrot(finger_2_pos, finger_2_rotmat)
    target_conf_2_list = []
    approach_conf_2_list = []
    rgtjcenter = []
    rri = []
    rfor_grasp_object = []

    # 抓finger_1,gofa5先是悬停在离夹爪20cm高的地方，再抓住finger_1，抬起原路返回到20cm高处
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        lftgrasp_homo = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
        lft_jaw_center_homo = finger_1_homo.dot(lftgrasp_homo)
        gripper_s.lft.grip_at_with_jcpose(lft_jaw_center_homo[:3, 3], lft_jaw_center_homo[:3, :3], jaw_width)

        mg_jawwidth = .05
        m_rotmat = lft_jaw_center_homo[:3, :3]
        value = np.array([(-.053 - mg_jawwidth) / 2, 0, -.1982])
        m_pos = lft_jaw_center_homo[:3, 3] + lft_jaw_center_homo[:3, :3].dot(value)
        gripper_s.fix_to(m_pos, m_rotmat)
        gripper_s.mg_jaw_to(mg_jawwidth)

        l_rotmat = jaw_center_rotmat.T
        laaa_m_pos = jaw_center_pos + jaw_center_rotmat.dot(value)
        l_pos = np.dot(jaw_center_rotmat.T, (-laaa_m_pos))
        try:
            target_conf = rbt_s.ik(component_name="arm",
                                   tgt_pos=m_pos,
                                   tgt_rotmat=m_rotmat,
                                   seed_jnt_values=rbt_s.get_jnt_values('arm'),
                                   max_niter=200,
                                   tcp_jnt_id=None,
                                   tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   local_minima="end",
                                   toggle_debug=False)
            rbt_s.fk('arm', target_conf)
            pos_1, rotmat_1 = rbt_s.get_gl_tcp('arm')
            if rbt_s.is_collided():
                pass
            else:
                target_conf_1_list.append(target_conf)
                pos, rot = rbt_s.get_gl_tcp("arm")
                lftapp_pos = pos + [0, 0, 0.2]
                try:
                    approach_conf = rbt_s.ik('arm', lftapp_pos, rot, seed_jnt_values=target_conf)
                    if approach_conf is not None:
                        approach_conf_1_list.append([approach_conf, target_conf])
                        rbt_s.fk('arm', approach_conf)
                        pos_2, rotmat_2 = rbt_s.get_gl_tcp('arm')
                        lftjcenter.append((l_pos, l_rotmat, pos_1, rotmat_1))
                        lli.append((jaw_center_pos, jaw_center_rotmat))
                        lfor_grasp_object.append((jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat))
                except:
                    pass
        except:
            pass

    # 抓finger_2,接上一步，gofa5从finger_1抬高20cm的地方，来到离finger_2抬高20cm的地方，再抓住finger_2
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        rgtgrasp_homo = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
        rgt_jaw_center_homo = finger_2_homo.dot(rgtgrasp_homo)
        gripper_s.rgt.grip_at_with_jcpose(rgt_jaw_center_homo[:3, 3], rgt_jaw_center_homo[:3, :3], jaw_width)

        mg_jawwidth = .05
        m_rotmat = rgt_jaw_center_homo[:3, :3]
        value = np.array([(.053 + mg_jawwidth) / 2, 0, -.1982])
        m_pos = rgt_jaw_center_homo[:3, 3] + rgt_jaw_center_homo[:3, :3].dot(value)
        gripper_s.fix_to(m_pos, m_rotmat)
        gripper_s.mg_jaw_to(mg_jawwidth)

        r_rotmat = jaw_center_rotmat.T
        raaa_m_pos = jaw_center_pos + jaw_center_rotmat.dot(value)
        r_pos = np.dot(jaw_center_rotmat.T, (-raaa_m_pos))

        try:
            target_conf = rbt_s.ik(component_name="arm",
                                   tgt_pos=m_pos,
                                   tgt_rotmat=m_rotmat,
                                   seed_jnt_values=rbt_s.get_jnt_values('arm'),
                                   max_niter=200,
                                   tcp_jnt_id=None,
                                   tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   local_minima="end",
                                   toggle_debug=False)
            rbt_s.fk('arm', target_conf)
            pos_1, rotmat_1 = rbt_s.get_gl_tcp('arm')
            if rbt_s.is_collided():
                pass
            else:
                target_conf_2_list.append(target_conf)
                pos, rot = rbt_s.get_gl_tcp("arm")
                rgtapp_pos = pos + [0, 0, 0.2]
                try:
                    approach_conf = rbt_s.ik('arm', rgtapp_pos, rot, seed_jnt_values=target_conf)
                    if approach_conf is not None:
                        approach_conf_2_list.append([approach_conf, target_conf])
                        rbt_s.fk('arm', approach_conf)
                        pos_2, rotmat_2 = rbt_s.get_gl_tcp('arm')
                        rgtjcenter.append((r_pos, r_rotmat, pos_1, rotmat_1))
                        rri.append((jaw_center_pos, jaw_center_rotmat))
                        rfor_grasp_object.append((jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat))
                except:
                    pass
        except:
            pass

    # init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # end_jnts = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # graph = defaultdict(dict)
    # for a1, _ in approach_conf_1_list:
    #     graph[tuple(init_jnts)][tuple(a1)] = 1
    #
    # for a1, t1 in approach_conf_1_list:
    #     graph[tuple(a1)][tuple(t1)] = 1
    #
    # for i, (_, t1) in enumerate(approach_conf_1_list):
    #     for j, (a2, _) in enumerate(approach_conf_2_list):
    #         graph[tuple(t1)][tuple(a2)] = cost(lftjcenter[i], rgtjcenter[j])
    #
    # for a2, t2 in approach_conf_2_list:
    #     graph[tuple(a2)][tuple(t2)] = 1
    #
    # for _, t2 in approach_conf_2_list:
    #     graph[tuple(t2)][tuple(end_jnts)] = 1

    init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_jnts = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    graph = defaultdict(dict)
    edge_info = {}

    for i, (a1, _) in enumerate(approach_conf_1_list):
        graph[tuple(init_jnts)][tuple(a1)] = 1
        edge_info[(tuple(init_jnts), tuple(a1))] = (None, None)

    for i, (a1, t1) in enumerate(approach_conf_1_list):
        graph[tuple(a1)][tuple(t1)] = 1
        edge_info[(tuple(a1), tuple(t1))] = (lftjcenter[i], None)  # 左手抓取姿态记录

    for i, (_, t1) in enumerate(approach_conf_1_list):
        for j, (a2, _) in enumerate(approach_conf_2_list):
            cost_val = cost(lftjcenter[i], rgtjcenter[j])
            graph[tuple(t1)][tuple(a2)] = cost_val
            edge_info[(tuple(t1), tuple(a2))] = (lftjcenter[i], rgtjcenter[j])  # 两手姿态配对

    for j, (a2, t2) in enumerate(approach_conf_2_list):
        graph[tuple(a2)][tuple(t2)] = 1
        edge_info[(tuple(a2), tuple(t2))] = (None, rgtjcenter[j])  # 右手抓取姿态记录

    for j, (_, t2) in enumerate(approach_conf_2_list):
        graph[tuple(t2)][tuple(end_jnts)] = 1
        edge_info[(tuple(t2), tuple(end_jnts))] = (None, None)

    path, center_pairs = dijkstra(graph, tuple(init_jnts), tuple(end_jnts), edge_info=edge_info)
    start, a1, g1, a2, g2, end = path

    for idx, (left_pose, right_pose) in enumerate(center_pairs):
        print(f"Edge {idx}:")
        print("  Left pose :", left_pose)
        print("  Right pose:", right_pose)

    rrtc_s = rrtc.RRTConnect(rbt_s)
    path_app_finger_1 = rrtc_s.plan('arm',
                                    init_jnts,
                                    np.array(a1),
                                    obstacle_list=[finger_1],
                                    otherrobot_list=[],
                                    ext_dist=0.03,
                                    max_iter=300,
                                    max_time=15.0,
                                    smoothing_iterations=50,
                                    animation=False)

    path_gri_finger_1 = rrtc_s.plan('arm',
                                    np.array(a1),
                                    np.array(g1),
                                    obstacle_list=[finger_1],
                                    otherrobot_list=[],
                                    ext_dist=0.03,
                                    max_iter=300,
                                    max_time=15.0,
                                    smoothing_iterations=50,
                                    animation=False)

    path_app_finger_2 = rrtc_s.plan('arm',  # 第一步：规划从finger_1抬高2ocm位姿到接近finger_2位姿的路径
                                    np.array(a1),
                                    np.array(a2),
                                    obstacle_list=[finger_1],
                                    otherrobot_list=[],
                                    ext_dist=0.03,
                                    max_iter=300,
                                    max_time=15.0,
                                    smoothing_iterations=50,
                                    animation=False)

    path_gri_finger_2 = rrtc_s.plan('arm',  # 若第一步可行，再第二步：规划从接近位姿到最终抓取位姿
                                    np.array(a2),
                                    np.array(g2),
                                    obstacle_list=[finger_1],
                                    otherrobot_list=[],
                                    ext_dist=0.03,
                                    max_iter=300,
                                    max_time=15.0,
                                    smoothing_iterations=50,
                                    animation=False)

    full_path = (path_app_finger_1 +        # 1. 接近finger_1
                 path_gri_finger_1 +        # 2. 抓finger_1
                 path_gri_finger_1[::-1] +  # 3. 原路返回到接近finger_1的位置
                 path_app_finger_2 +        # 4. 接近finger_2
                 path_gri_finger_2 +        # 5. 抓finger_2
                 path_gri_finger_2[::-1])   # 6. 原路返回到接近finger_2的位置

    mg_jawwidth_list = (np.linspace(0, mg_jawwidth, len(path_app_finger_1)).tolist() +
                        [mg_jawwidth] * len(path_gri_finger_1) +
                        [mg_jawwidth] * len(path_gri_finger_1) +
                        [mg_jawwidth] * len(path_app_finger_2) +
                        [mg_jawwidth] * len(path_gri_finger_2) +
                        [mg_jawwidth] * len(path_gri_finger_2))
    lft_jawwidth = .012
    lft_jawwidth_list = ([0] * len(path_app_finger_1) +
                         np.linspace(0, lft_jawwidth, len(path_gri_finger_1)).tolist() +
                         [lft_jawwidth] * len(path_gri_finger_1) +
                         [lft_jawwidth] * len(path_app_finger_2) +
                         [lft_jawwidth] * len(path_gri_finger_2) +
                         [lft_jawwidth] * len(path_gri_finger_2))
    rgt_jawwidth = .012
    rgt_jawwidth_list = ([0] * len(path_app_finger_1) +
                         [0] * len(path_gri_finger_1) +
                         [0] * len(path_gri_finger_1) +
                         [0] * len(path_app_finger_2) +
                         np.linspace(0, rgt_jawwidth, len(path_gri_finger_2)).tolist() +
                         [rgt_jawwidth] * len(path_gri_finger_2))

    objpose_list_1 = []
    objpose_list_2 = []
    objpose_list_1 += [finger_1_homo for _ in range(len(full_path))]
    objpose_list_2 += [finger_2_homo for _ in range(len(full_path))]

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

