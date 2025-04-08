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
import robot_con.gofa_con.gofa_con as gofa_con
import robot_con.reconfgripper.maingripper.maingripper as dh_r


if __name__ == '__main__':
    # rbt_r = gofa_con.GoFaArmController(toggle_debug=False)   # 控制gofa5
    # dh76_con = dh_r.MainGripper(port="com4")                 # 控制dh
    # dh76_con.init_gripper()                                  # dh初始化

    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    obj_path = f"../../0000_examples/objects/finger_a.stl"
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
    # finger_1.attach_to(base)

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
    # finger_2.attach_to(base)

    rbt_s = gf5.GOFA5()
    gripper_s = hnd.reconfgripper(pos=rbt_s.arm.jnts[-1]['gl_posq'],
                                  rotmat=rbt_s.arm.jnts[-1]['gl_rotmatq'],
                                  name='hnd', enable_cc=False)
    grasp_info_list = gpa.load_pickle_file('finger', '../../0000_examples/', 'reconfgripper_finger_grasps.pickle')
    # rbt_s.gen_meshmodel().attach_to(base)

    finger_1_homo = rm.homomat_from_posrot(finger_1_pos, finger_1_rotmat)    # 将pos和rot转换为4×4齐次变换矩阵，rot默认I
    target_conf_1_list = []
    approach_conf_1_list = []

    finger_2_homo = rm.homomat_from_posrot(finger_2_pos, finger_2_rotmat)
    target_conf_2_list = []
    approach_conf_2_list = []
    i = 0

    lftjcenter = []
    aa = []
    # 抓finger_1,gofa5先是悬停在离夹爪20cm高的地方，再抓住finger_1，抬起原路返回到20cm高处
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        lftgrasp_homo = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)  # 合成4×4的齐次矩阵
        lft_jaw_center_homo = finger_1_homo.dot(lftgrasp_homo)                     # 初始抓取信息列表的obj在原点，要和本代码finger_1的位姿一致
        gripper_s.lft.grip_at_with_jcpose(lft_jaw_center_homo[:3, 3], lft_jaw_center_homo[:3, :3], jaw_width)  # 放lft

        mg_jawwidth = .05                                         # 放好lft，以此参考放整个dh夹爪
        m_rotmat = lft_jaw_center_homo[:3, :3]
        value = np.array([(-.053 - mg_jawwidth) / 2, 0, -.1982])  # dh60根部和lft小夹爪抓取中心的pos差值
        m_pos = lft_jaw_center_homo[:3, 3] + lft_jaw_center_homo[:3, :3].dot(value)
        gripper_s.fix_to(m_pos, m_rotmat)
        gripper_s.mg_jaw_to(mg_jawwidth)

        jaw_center_pos_in_m = np.dot(m_rotmat.T, (lft_jaw_center_homo[:3, 3] - m_pos))
        jaw_center_rotmat_in_m = np.dot(m_rotmat.T, lft_jaw_center_homo[:3, :3])
        try:
            target_conf = rbt_s.ik(component_name="arm",      # 机械臂求ik
                                   tgt_pos=m_pos,
                                   tgt_rotmat=m_rotmat,
                                   seed_jnt_values=rbt_s.get_jnt_values('arm'),
                                   max_niter=200,
                                   tcp_jnt_id=None,
                                   tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   local_minima="end",
                                   toggle_debug=False)
            rbt_s.fk('arm', target_conf)       # 机械臂执行fk到达目标位姿
            if rbt_s.is_collided():                           # 若碰撞，丢弃
                pass
            else:
                target_conf_1_list.append(target_conf)        # 若不碰撞，则将这个target_conf加入target_conf_1_list
                pos, rot = rbt_s.get_gl_tcp("arm")            # 获取机械臂工具中心点jaw_center的全局位姿
                lftapp_pos = pos + [0, 0, 0.2]                # finger_1抬高20cm的位置，用于接近目标
                try:
                    approach_conf = rbt_s.ik('arm', lftapp_pos, rot, seed_jnt_values=target_conf)  # 计算接近目标的机械臂关节角度
                    approach_conf_1_list.append([approach_conf, target_conf])     # 将接近目标信息和目标信息一起存储
                    rbt_s.fk('arm', approach_conf)                 # 机械臂执行fk到达接近目标位姿
                    lftjcenter.append((jaw_center_pos_in_m, jaw_center_rotmat_in_m))
                    aa.append((jaw_center_pos, jaw_center_rotmat))
                except:
                    pass
        except:
            pass

    init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rrtc_s = rrtc.RRTConnect(rbt_s)

    lftcount = -1
    for approach_conf in approach_conf_1_list:
        path_app_finger_1 = rrtc_s.plan('arm',  # 第一步：规划从起始位姿到接近位姿的路径
                                        init_jnts,
                                        approach_conf[0],
                                        obstacle_list=[finger_1],
                                        otherrobot_list=[],
                                        ext_dist=0.03,
                                        max_iter=300,
                                        max_time=15.0,
                                        smoothing_iterations=50,
                                        animation=False)
        lftcount += 1
        if path_app_finger_1 is not None:
            print(f"从初始到接近位置可以得到路径")
            path_gri_finger_1 = rrtc_s.plan('arm',  # 若第一步可行，再第二步：规划从接近位姿到最终抓取位姿
                                            approach_conf[0],
                                            approach_conf[1],
                                            obstacle_list=[finger_1],
                                            otherrobot_list=[],
                                            ext_dist=0.03,
                                            max_iter=300,
                                            max_time=15.0,
                                            smoothing_iterations=50,
                                            animation=False)
            if path_gri_finger_1 is not None:
                print(f"从接近位置到目标位置可以得到路径")
                rbt_s.fk('arm', approach_conf[0])
                rbt_s.hnd.mg_jaw_to(mg_jawwidth)
                # rbt_s.gen_meshmodel().attach_to(base)
                init_2_jnts = approach_conf[0]

                rbt_s.fk('arm', approach_conf[1])
                rbt_s.hnd.mg_jaw_to(mg_jawwidth)
                # rbt_s.gen_meshmodel().attach_to(base)
                break

    rgtjcenter = []
    bb = []
    # 抓finger_2,接上一步，gofa5从finger_1抬高20cm的地方，来到离finger_2抬高20cm的地方，再抓住finger_2
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        rgtgrasp_homo = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)  # 合成4×4的齐次矩阵
        rgt_jaw_center_homo = finger_2_homo.dot(rgtgrasp_homo)                     # 初始抓取信息列表的obj在原点，要和本代码finger_2的位姿一致
        gripper_s.rgt.grip_at_with_jcpose(rgt_jaw_center_homo[:3, 3], rgt_jaw_center_homo[:3, :3], jaw_width)

        mg_jawwidth = .05                                         # 放好rgt，以此参考放整个dh夹爪
        m_rotmat = rgt_jaw_center_homo[:3, :3]
        value = np.array([(.053 + mg_jawwidth) / 2, 0, -.1982])  # dh60根部和rgt小夹爪抓取中心的pos差值
        m_pos = rgt_jaw_center_homo[:3, 3] + rgt_jaw_center_homo[:3, :3].dot(value)
        gripper_s.fix_to(m_pos, m_rotmat)
        gripper_s.mg_jaw_to(mg_jawwidth)

        jaw_center_pos_in_m = np.dot(m_rotmat.T, (rgt_jaw_center_homo[:3, 3] - m_pos))
        jaw_center_rotmat_in_m = np.dot(m_rotmat.T, rgt_jaw_center_homo[:3, :3])
        try:
            target_conf = rbt_s.ik(component_name="arm",      # 机械臂求ik
                                   tgt_pos=m_pos,
                                   tgt_rotmat=m_rotmat,
                                   seed_jnt_values=rbt_s.get_jnt_values('arm'),
                                   max_niter=200,
                                   tcp_jnt_id=None,
                                   tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   local_minima="end",
                                   toggle_debug=False)
            rbt_s.fk('arm', target_conf)       # 机械臂执行fk到达目标位姿
            if rbt_s.is_collided():                           # 若碰撞，丢弃
                pass
            else:
                target_conf_2_list.append(target_conf)          # 若不碰撞，则将这个target_conf加入target_conf_2_list
                pos, rot = rbt_s.get_gl_tcp("arm")            # 获取机械臂工具中心点jaw_center的全局位姿
                rgtapp_pos = pos + [0, 0, 0.2]                # finger_2抬高20cm的位置，用于接近目标
                try:
                    approach_conf = rbt_s.ik('arm', rgtapp_pos, rot, seed_jnt_values=target_conf)  # 计算接近目标的机械臂关节角度
                    approach_conf_2_list.append([approach_conf, target_conf])     # 将接近目标信息和目标信息一起存储
                    rbt_s.fk('arm', approach_conf)               # 机械臂执行fk到达接近目标位姿
                    rgtjcenter.append((jaw_center_pos_in_m, jaw_center_rotmat_in_m))
                    bb.append((jaw_center_pos, jaw_center_rotmat))
                except:
                    pass
        except:
            pass

    rgtcount = -1
    for approach_conf in approach_conf_2_list:
        path_app_finger_2 = rrtc_s.plan('arm',  # 第一步：规划从finger_1抬高2ocm位姿到接近finger_2位姿的路径
                                        init_2_jnts,
                                        approach_conf[0],
                                        obstacle_list=[finger_1],
                                        otherrobot_list=[],
                                        ext_dist=0.03,
                                        max_iter=300,
                                        max_time=15.0,
                                        smoothing_iterations=50,
                                        animation=False)
        rgtcount += 1
        if path_app_finger_2 is not None:
            path_gri_finger_2 = rrtc_s.plan('arm',  # 若第一步可行，再第二步：规划从接近位姿到最终抓取位姿
                                            approach_conf[0],
                                            approach_conf[1],
                                            obstacle_list=[finger_1],
                                            otherrobot_list=[],
                                            ext_dist=0.03,
                                            max_iter=300,
                                            max_time=15.0,
                                            smoothing_iterations=50,
                                            animation=False)
            if path_gri_finger_2 is not None:
                rbt_s.fk('arm', approach_conf[0])
                rbt_s.hnd.mg_jaw_to(mg_jawwidth)
                # rbt_s.gen_meshmodel().attach_to(base)

                rbt_s.fk('arm', approach_conf[1])
                rbt_s.hnd.mg_jaw_to(mg_jawwidth)
                # rbt_s.hnd.rgt.jaw_to(jaw_width)
                # rbt_s.gen_meshmodel().attach_to(base)
                break

    def update(robot_s,                    # 机器人模型
               object_box,                 # 物体碰撞模型，用于动画中显示物体的位置变化
               robot_path,                 # 机器人的运动轨迹列表，每个元素是一个关节角度数组（conf_list）
               jawwidth_path,              # 夹爪的开合轨迹列表，每个元素是一个数值（jawwidth_list）
               lft_jawwidth_list,
               rgt_jawwidth_list,
               lftobj_path,                   # 物体的运动轨迹列表，每个元素是一个4×4矩阵（objpose_list）
               rgtobj_path,
               robot_attached_list,        # 记录机器人上一帧的3D模型，在新一帧时删除，以免重叠
               object_attached_list,       # 记录物体上一帧的3D模型，在新一帧时删除，以免重叠
               counter,                    # 计数器（列表 counter=[0]），跟踪当前播放到哪一帧
               task):                      # Panda3D的任务对象，用于控制循环动画

        if counter[0] >= len(robot_path):  # 如果到达路径终点，则循环回到起点
            counter[0] = 0
        if len(robot_attached_list) != 0:  # 清除上一帧的机器人和物体模型
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()

        # 更新机器人位置
        pose = robot_path[counter[0]]                          # 取当前帧的关节角度
        robot_s.fk("arm", pose)                                # 让机器人运动到该位置
        robot_s.hnd.mg_jaw_to(jawwidth_path[counter[0]])       # 设置dh夹爪的开合
        robot_s.hnd.lft.jaw_to(lft_jawwidth_list[counter[0]])  # 设置lft的开合
        robot_s.hnd.rgt.jaw_to(rgt_jawwidth_list[counter[0]])  # 设置rgt的开合

        robot_meshmodel = robot_s.gen_meshmodel()              # 生成机器人3D模型
        robot_meshmodel.attach_to(base)                        # 把机器人模型添加到场景
        robot_attached_list.append(robot_meshmodel)            # 记录当前帧的机器人模型

        # 更新手指1位置
        lftobj_pose = lftobj_path[counter[0]]                     # 取当前帧的物体位姿
        lftobjb_copy = object_box.copy()                          # 复制物体模型
        lftobjb_copy.set_homomat(lftobj_pose)                     # 设置新的位姿
        lftobjb_copy.attach_to(base)                              # 把物体添加到场景
        object_attached_list.append(lftobjb_copy)                 # 记录当前帧的物体模型

        # 更新手指2位置
        rgtobj_pose = rgtobj_path[counter[0]]
        rgtobjb_copy = object_box.copy()
        rgtobjb_copy.set_homomat(rgtobj_pose)
        rgtobjb_copy.attach_to(base)
        object_attached_list.append(rgtobjb_copy)
        counter[0] += 1
        return task.again                                         # 让任务继续执行

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

    # 第一个手指的运动轨迹
    lpos, lrotmat = lftjcenter[lftcount]
    aajaw_center_pos, aajaw_center_rotmat = aa[rgtcount]
    print(lpos)
    print(aajaw_center_pos)
    objpose_list_1 = []
    objpose_list_1 += [finger_1_homo for _ in range(len(path_app_finger_1))]
    objpose_list_1 += [finger_1_homo for _ in range(len(path_gri_finger_1))]

    for i in range(len(path_gri_finger_1)):
        jaw_pose = path_gri_finger_1[::-1][i]
        rbt_s.fk("arm", jaw_pose)
        pos, rotmat = rbt_s.get_gl_tcp('arm')

        f_rotmat = rotmat.dot(aajaw_center_rotmat.T)
        f_pos = pos + np.dot(rotmat, lpos)
        ee_pose = rm.homomat_from_posrot(f_pos, f_rotmat)
        objpose_list_1.append(ee_pose)

    for i in range(len(path_app_finger_2)):
        jaw_pose = path_app_finger_2[i]
        rbt_s.fk("arm", jaw_pose)
        pos, rotmat = rbt_s.get_gl_tcp('arm')

        f_rotmat = rotmat.dot(aajaw_center_rotmat.T)
        f_pos = pos + np.dot(rotmat, lpos)
        ee_pose = rm.homomat_from_posrot(f_pos, f_rotmat)
        objpose_list_1.append(ee_pose)

    for i in range(len(path_gri_finger_2)):
        jaw_pose = path_gri_finger_2[i]
        rbt_s.fk("arm", jaw_pose)
        pos, rotmat = rbt_s.get_gl_tcp('arm')

        f_rotmat = rotmat.dot(aajaw_center_rotmat.T)
        f_pos = pos + np.dot(rotmat, lpos)
        ee_pose = rm.homomat_from_posrot(f_pos, f_rotmat)
        objpose_list_1.append(ee_pose)

    for i in range(len(path_gri_finger_2)):
        jaw_pose = path_gri_finger_2[::-1][i]
        rbt_s.fk("arm", jaw_pose)
        pos, rotmat = rbt_s.get_gl_tcp('arm')

        f_rotmat = rotmat.dot(aajaw_center_rotmat.T)
        f_pos = pos + np.dot(rotmat, lpos)
        ee_pose = rm.homomat_from_posrot(f_pos, f_rotmat)
        objpose_list_1.append(ee_pose)

    # 第二个手指的运动轨迹
    rpos, rrotmat = rgtjcenter[rgtcount]
    bbjaw_center_pos, bbjaw_center_rotmat = bb[rgtcount]

    objpose_list_2 = []
    objpose_list_2 += [finger_2_homo for _ in range(len(path_app_finger_1))]
    objpose_list_2 += [finger_2_homo for _ in range(len(path_gri_finger_1))]
    objpose_list_2 += [finger_2_homo for _ in range(len(path_gri_finger_1))]
    objpose_list_2 += [finger_2_homo for _ in range(len(path_app_finger_2))]
    objpose_list_2 += [finger_2_homo for _ in range(len(path_gri_finger_2))]

    for i in range(len(path_gri_finger_2)):
        jaw_pose = path_gri_finger_2[::-1][i]
        rbt_s.fk("arm", jaw_pose)
        pos, rotmat = rbt_s.get_gl_tcp('arm')

        f_rotmat = rotmat.dot(bbjaw_center_rotmat.T)
        f_pos = pos + np.dot(rotmat, rpos)
        ee_pose = rm.homomat_from_posrot(f_pos, f_rotmat)
        objpose_list_2.append(ee_pose)

    robot_attached_list = []
    object_attached_list = []
    counter = [0]

    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbt_s,
                                     finger_1,
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