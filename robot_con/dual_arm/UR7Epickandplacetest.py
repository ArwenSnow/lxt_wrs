import math
import time
# import robot_sim.robots.ur7e.ur7e_withouttable as ur7
# import robot_con.ur.ur7_dh50_rtde as ur7con
import basis.robot_math as rm
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import visualization.panda.world as wd
import robot_sim.robots.XME3P_dual.XME3P_dual as Xme
import modeling.geometric_model as gm
import manipulation.pick_place_planner_1 as ppp
import motion.probabilistic.rrt_connect as rrtc
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh50.dh50 as hnd
import socket
import threading
import struct
import pickle


if __name__ == '__main__':
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    gm.gen_frame().attach_to(base)

    rbt_s = Xme.XME3P_dual()
    # rbt_s.gen_meshmodel().attach_to(base)
    hnd_s = hnd.Dh50()

    table = cm.CollisionModel("object/wholetable.stl", cdprimit_type="box", expand_radius=-.003)
    table.set_rgba([0.35, 0.35, 0.35, 1])
    table.attach_to(base)
    workpiece_before = cm.CollisionModel("object/box.stl", cdprimit_type="box", expand_radius=.001)
    workpiece_before.set_pos(pos=np.array([0.8, -0.2, 0.0]))
    workpiece_after = cm.CollisionModel("object/box.stl", cdprimit_type="box", expand_radius=.001)
    workpiece_after.set_pos(pos=np.array([0.8, -0.0, 0.3]))
    workpiece_final = cm.CollisionModel("object/box.stl", cdprimit_type="box", expand_radius=.001)
    workpiece_final.set_pos(pos=np.array([0.8, 0.2, 0.2]))

    obgl_start_homomat = rm.homomat_from_posrot(workpiece_before.get_pos(), np.eye(3))
    obgl_goal_homomat = rm.homomat_from_posrot(workpiece_after.get_pos(), np.eye(3))
    obgl_final_homomat = rm.homomat_from_posrot(workpiece_final.get_pos(), np.eye(3))

    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)

    original_grasp_info_list = gpa.load_pickle_file('box', './', 'dh76.pickle')

    lft_manipulator_name = "lft_arm"
    lft_hand_name = "lft_hnd"
    lft_component_name = "lft_arm"
    rgt_manipulator_name = "rgt_arm"
    rgt_hand_name = "rgt_hnd"
    rgt_component_name = "rgt_arm"

    obstacle_list = [table]
    start_conf = rbt_s.get_jnt_values(lft_manipulator_name)
    conf_list, jawwidth_list, objpose_list, conf_list_approach, conf_list_middle, conf_list_depart = \
        ppp_s.gen_pick_and_place_motion(component_name=lft_component_name,
                                        hnd_name=lft_hand_name,
                                        objcm=workpiece_before,
                                        grasp_info_list=original_grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        obstacle_list=obstacle_list,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[np.array([0, 0, -1]), np.array([0, 0, -1])],
                                        approach_distance_list=[.0] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), np.array([0, 0, 1])],
                                        depart_distance_list=[.0] * 2)

    rbt_s.fk(component_name='lft_arm', jnt_values=conf_list_middle[-1])
    rbt_s.jaw_to(jawwidth_list[0:len(conf_list_approach+conf_list_middle)][-1])
    # rbt_s.gen_meshmodel().attach_to(base)

    conf_list2, jawwidth_list2, objpose_list2, conf_list_approach2, conf_list_middle2, conf_list_depart2 = \
        ppp_s.gen_pick_and_place_motion(component_name=rgt_component_name,
                                        hnd_name=rgt_hand_name,
                                        objcm=workpiece_before,
                                        grasp_info_list=original_grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        obstacle_list=obstacle_list,
                                        goal_homomat_list=[obgl_goal_homomat, obgl_final_homomat],
                                        approach_direction_list=[np.array([0, 0, 1]), np.array([0, 0, 0])],
                                        approach_distance_list=[.0] * 2,
                                        depart_direction_list=[np.array([0, 0, -1]), np.array([0, 0, 0])],
                                        depart_distance_list=[.0] * 2)

    robot_attached_list = []
    object_attached_list = []
    counter = [0]

    robotpath = []
    robotpath1 = conf_list_approach+conf_list_middle
    robotpath1_14 = np.array([list(item) + list(start_conf) for item in robotpath1])
    robotpath2 = conf_list_approach2+conf_list_middle2
    conf_last = np.array(conf_list_middle[-1])  # shape: (6,)
    robotpath2_array = np.array(robotpath2)  # shape: (n, 6)
    robotpath2_14 = np.hstack([np.tile(conf_last, (robotpath2_array.shape[0], 1)), robotpath2_array])
    robotpath.extend(robotpath1_14)
    robotpath.extend(robotpath2_14)

    robot_jawwidth_list = []
    jawwidth_list = jawwidth_list[0:len(robotpath1)]
    conf_last = jawwidth_list2[1]
    robot_jawwidth_list1_2 = [[element, conf_last] for element in jawwidth_list]

    conf_last = jawwidth_list[-1]
    robot_jawwidth_list2_2 = [[conf_last, element] for element in jawwidth_list2]
    robot_jawwidth_list.extend(robot_jawwidth_list1_2)
    robot_jawwidth_list.extend(robot_jawwidth_list2_2)

    objpose_list = objpose_list[0:len(robotpath1)]
    objpose_list = objpose_list+objpose_list2

    paths = {
        "path1": robotpath1_14,
        "path2": robotpath2_14,
    }

    with open("path.pickle", "wb") as f:
        pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 发送到c++端
    SEND_IP_1 = '169.254.160.100'
    SEND_PORT_1 = 10000
    SEND_IP_2 = '192.168.0.100'
    SEND_PORT_2 = 10001

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr_1 = (SEND_IP_1, SEND_PORT_1)
    server_addr_2 = (SEND_IP_2, SEND_PORT_2)


    def update(robot_s,
               object_box,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter,
               task):

        # 生成动画
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        pose = robot_path[counter[0]]
        robot_s.fk('both', pose)

        robot_s.jaw_to(hand_name='both', both_jawwidth=robot_jawwidth_list[counter[0]])

        robot_meshmodel = robot_s.gen_meshmodel(is_machine=True)
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)

        obj_pose = obj_path[counter[0]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([1, 0, 0, 1])
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)

        counter[0] += 1

        return task.again

    object_holder = workpiece_before
    taskMgr.doMethodLater(0.04, update, "update",
                          extraArgs=[rbt_s,
                                     object_holder,
                                     robotpath,
                                     robot_jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)  # 0.01表示动画帧数（放映快慢），增大变慢，应该小于1/24才有连贯动画

    for i in robotpath1_14:
        try:
            msg_bytes = struct.pack('7f', *i[:7].astype(np.float32))  # 转 float32 发送
            sender.sendto(msg_bytes, server_addr_1)
            time.sleep(0.01)
            print("发送关节角度到c++端:", i)
            print("\n")
        except Exception as e:
            print(f"发送失败: {e}")

    for i in robotpath2_14:
        try:
            msg_bytes = struct.pack('7f', *i[7:].astype(np.float32))  # 转 float32 发送
            sender.sendto(msg_bytes, server_addr_2)
            time.sleep(0.01)
            print("发送关节角度到c++端:", i)
            print("\n")
        except Exception as e:
            print(f"发送失败: {e}")

    base.run()



