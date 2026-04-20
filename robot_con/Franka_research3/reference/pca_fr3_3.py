import time
import socket
import json
import numpy as np
import threading
import math
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.pca.pca as pca
import robot_sim.robots.Franka_research3.Franka_research3 as fr3

#!/usr/bin/env python3
# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE
import argparse
from pylibfranka import ControllerMode, JointPositions, Robot
import sys
sys.path.insert(0, '/home/lruiqing/wrs_shu/robot_con/libfranka/pylibfranka/examples')

from example_common import MotionGenerator


def make_homo(rotmat, tvec):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo

#    ===============守护线程1：接收主端数据===============
UDP_IP = '0.0.0.0'
UDP_PORT = 14000
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")

first_angles = None
latest_angles = None
prev_angles = None
latest_time = None
data_lock = threading.Lock()

def udp_receiver():
    global latest_angles, latest_time, first_angles
    while True:                                                         # UDP需要循环执行
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)
            # print("接收到角度：", msg)

            angles = np.array(msg)
            angles = np.deg2rad(angles)
            # print("转为弧度：", angles)

            with data_lock:                                             # 改数据时，主线程不能来读
                latest_angles = angles

            if first_angles is None:
                first_angles = angles.copy()
                print("第一次收到的角度（rad）：", first_angles)

        except Exception as e:
            print(f"UDP 接收错误: {e}")

recv_thread = threading.Thread(target=udp_receiver, daemon=True)       # 主线程一旦结束，守护线程也结束
recv_thread.start()

#    ===============wrs仿真===============
base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
gm.gen_frame().attach_to(base)
robot_1 = pca.Pca()
robot_1.fk('arm', jnt_values=first_angles)
p_1, r_1 = robot_1.get_gl_tcp('arm')
init_1 = make_homo(r_1, p_1)
current_robot_mesh = robot_1.gen_meshmodel(toggle_tcpcs=True)
current_robot_mesh.attach_to(base)

robot_2 = fr3.Franka_research3()
fr3_pos = np.array([.50682, .6, 0])
fr3_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
robot_2.fix_to(fr3_pos, fr3_rot)
T = None
current_robot_mesh_2 = None

T_ready = threading.Event()
first_target_event = threading.Event()  # 用于通知franka控制线程第一个目标已就绪
def franka_control_thread():
    global T, current_robot_mesh_2
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()
    robot = Robot(args.ip)

    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )
        active_control = robot.start_joint_position_control(ControllerMode.CartesianImpedance)
        robot_state, duration = active_control.readOnce()
        initial_position = robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q
        robot_2.fk("arm", np.array(initial_position))
        current_robot_mesh_2 = robot_2.gen_meshmodel(toggle_tcpcs=True)
        current_robot_mesh_2.attach_to(base)

        p_2, r_2 = robot_2.get_gl_tcp('arm')
        init_2 = make_homo(r_2, p_2)
        T = init_2 @ np.linalg.inv(init_1)
        print("初始位置：", initial_position)
        print("遥操作到fr3的转换关系是：", T)
        T_ready.set()

        motion_start = False
        motion_finished = False
        time_elapsed = 0.0
        id = 0
        speed_factor = 0.5  # 可
        first_target_event.wait()
        print("已收到第一个目标，开始控制")
        recorded_positions = []

        # with fr3_target_lock:
        #     first_target = fr3_conf_new.copy()
        # motion_gen = MotionGenerator(speed_factor, first_target)
        speed_factor = 0.2

        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            time_elapsed += duration.to_sec()

            if id <10:
                new_positions = initial_position
                joint_positions = JointPositions(new_positions)
                id += 1
            else:
                with fr3_target_lock:
                    motion_gen = MotionGenerator(speed_factor, fr3_conf_new)
                    joint_positions = motion_gen(robot_state, duration.to_sec())

            # joint_positions = JointPositions(new_positions)
            print("当前时间为：", time_elapsed, "，当前关节角度为：", joint_positions.q)

            # Send command to robot
            active_control.writeOnce(joint_positions)
            recorded_positions.append(np.array(joint_positions.q).copy())


    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
    finally:
        if robot is not None:
            robot.stop()
        # 保存记录的数据
        if recorded_positions:
            np.savetxt("recorded_positions.txt", recorded_positions, delimiter=',')
            print("已保存 recorded_positions.txt")

control_thread = threading.Thread(target=franka_control_thread, daemon=True)
control_thread.start()


def update_robot_joints_1(robot, jnt_values):
    global current_robot_mesh
    try:
        if current_robot_mesh is not None:
            current_robot_mesh.detach()

        robot.fk(jnt_values=jnt_values)
        p, r = robot.get_gl_tcp('arm')
        new_2 = T @ make_homo(r, p)
        new_p, new_r = new_2[:3, 3], new_2[:3, :3]
        new_mesh = robot.gen_meshmodel()
        new_mesh.attach_to(base)
        current_robot_mesh = new_mesh
        return new_p, new_r
    except Exception as e:
        print(f"更新意优机器人失败: {e}")


def update_robot_joints_2(pos, rotmat):
    """根据关节角度更新机器人状态"""
    global current_robot_mesh_2
    try:
        conf = robot_2.ik(component_name='arm', tgt_pos=pos, tgt_rotmat=rotmat)
        robot_2.fk(jnt_values=conf)
        if conf is None or (isinstance(conf, np.ndarray) and conf.size == 0):
            print("franka机器人无法解ik")

        # 移除旧的机器人模型
        if current_robot_mesh_2 is not None:
            current_robot_mesh_2.detach()

        # 更新机器人关节角度（前向运动学）
        robot_2.fk(jnt_values=conf)

        # 生成新的机器人模型
        new_mesh = robot_2.gen_meshmodel()
        new_mesh.attach_to(base)
        current_robot_mesh_2 = new_mesh

        return conf

    except Exception as e:
        print(f"更新franka机器人时出错: {e}")
        return False

prev_fr3_conf = None
diff = np.array([0, 0, 0, 0, 0, 0, 0])
fr3_target_lock = threading.Lock()
fr3_conf_new = None

#    ===============主线程：计算映射===============
def update_task(task):
    global prev_angles, prev_fr3_conf, diff, fr3_conf_new
    if not T_ready.is_set():
        return task.again
    with data_lock:
        if latest_angles is None:
            return task.again
        angles_new = latest_angles.copy()
        delay = latest_time

    if prev_angles is None:                                             # 第一次没有上一帧，直接显示
        angles_display = angles_new
    else:
        alpha = 0.1                                                     # 线性插值，可调节平滑程度
        angles_display = (1-alpha) * prev_angles + alpha * angles_new

    new_p_2, new_r_2 = update_robot_joints_1(robot_1, angles_display)
    fr3_conf = update_robot_joints_2(new_p_2, new_r_2)

    with fr3_target_lock:
        fr3_conf_new = fr3_conf.copy()
        # fr3_conf_new[0] += 0.01908966
        # fr3_conf_new[2] -= 0.015765362648328684
        # fr3_conf_new[4] -=  0.004195
        # fr3_conf_new[6] += 0.003865
        print(fr3_conf_new)
    first_target_event.set()

    prev_fr3_conf = fr3_conf  # 更新上一次值

    prev_angles = angles_display.copy()

    return task.again


taskMgr.doMethodLater(0.001, update_task, "robot_update")
try:
    base.run()
except KeyboardInterrupt:
    print("程序退出")
    server.close()