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
from pylibfranka import ControllerMode, JointPositions, Robot, Torques
from robot_con.Franka_research3.teleoperation.joint_impedance_controller import JointImpedanceController

# ================ 辅助函数 =================
def make_homo(rotmat, tvec):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo

# ================ UDP 接收线程 =================
UDP_IP = '0.0.0.0'
UDP_PORT = 14000
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")

first_angles = None
latest_angles = None
prev_angles = None
data_lock = threading.Lock()

def udp_receiver():
    global latest_angles, first_angles
    while True:
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)
            # print("接收到角度：", msg)
            angles = np.array(msg)
            angles = np.deg2rad(angles)
            # print("转为弧度：", angles)

            with data_lock:
                latest_angles = angles

            if first_angles is None:
                first_angles = angles.copy()
                print("第一次收到的角度（rad）：", first_angles)
        except Exception as e:
            print(f"UDP 接收错误: {e}")

recv_thread = threading.Thread(target=udp_receiver, daemon=True)
recv_thread.start()

# ================ 机器人初始化 =================
import robot_sim.robots.pca.pca as pca
import robot_sim.robots.Franka_research3.Franka_research3 as fr3

robot_1 = pca.Pca()
robot_1.fk('arm', jnt_values=first_angles)
p_1, r_1 = robot_1.get_gl_tcp('arm')
init_1 = make_homo(r_1, p_1)

robot_2 = fr3.Franka_research3()
T = None
T_ready = threading.Event()
first_target_event = threading.Event()

control_count = 0
plan_count = 0
count_lock = threading.Lock()

# ================ Franka 控制线程 =================
def franka_control_thread():
    global T, control_count
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()
    robot = Robot(args.ip)

    try:
        robot.set_collision_behavior(
            [100.0]*7, [100.0]*7, [100.0]*6, [100.0]*6
        )
        active_control = robot.start_torque_control()
        robot_state, _ = active_control.readOnce()
        initial_position = robot_state.q if not hasattr(robot_state, "q_d") else robot_state.q_d

        robot_2.fk("arm", np.array(initial_position))
        p_2, r_2 = robot_2.get_gl_tcp('arm')
        init_2 = make_homo(r_2, p_2)
        T = init_2 @ np.linalg.inv(init_1)
        print("franka机器人初始位置：", initial_position)
        print("遥操作到fr3的转换关系是：", T)
        T_ready.set()

        # 阻抗控制器参数
        joint_stiffness = [50.0] * 7
        joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]
        controller = JointImpedanceController(
            stiffness=joint_stiffness,
            damping=joint_damping,
            alpha=0.2,
            speed_factor=0.2,
            max_target_age=0.5
        )
        model = robot.load_model()

        # 等待第一个目标
        first_target_event.wait()
        print("已收到第一个目标，开始控制")

        time_elapsed = 0.0
        while True:
            with count_lock:
                control_count += 1
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            current_time = time.time()
            time_elapsed += dt

            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            with fr3_target_lock:
                if fr3_conf_new is not None:
                    controller.set_target(fr3_conf_new.copy(), current_time)

            tau = controller.update(q, dq, dt, current_time)
            coriolis = np.array(model.coriolis(robot_state))
            tau_desired = tau + coriolis

            torque_command = Torques(tau_desired.tolist())
            active_control.writeOnce(torque_command)
            # print(f"当前时间: {time_elapsed:.3f}, 力矩: {tau_desired}")

    except Exception as e:
        print(f"控制线程错误: {e}")
    finally:
        robot.stop()

# ================ 映射计算线程 =================
def update_robot_joints_1(robot, jnt_values):
    try:
        robot.fk(jnt_values=jnt_values)
        p, r = robot.get_gl_tcp('arm')
        new_2 = T @ make_homo(r, p)
        new_p, new_r = new_2[:3, 3], new_2[:3, :3]
        return new_p, new_r
    except Exception as e:
        print(f"更新意优机器人失败: {e}")
        return None, None

def update_robot_joints_2(pos, rotmat):
    try:
        conf = robot_2.ik(component_name='arm', tgt_pos=pos, tgt_rotmat=rotmat)
        if conf is None or (isinstance(conf, np.ndarray) and conf.size == 0):
            print("franka机器人无法解ik")
            return None
        robot_2.fk(jnt_values=conf)
        return conf
    except Exception as e:
        print(f"更新franka机器人时出错: {e}")
        return None

prev_fr3_conf = None
fr3_target_lock = threading.Lock()
fr3_conf_new = None

# def update_task_loop():
#     global prev_angles, prev_fr3_conf, fr3_conf_new, plan_count
#     while True:
#         with count_lock:
#             plan_count += 1
#         loop_start = time.time()
#
#         if not T_ready.is_set():
#             time.sleep(0.001)
#             continue
#
#         with data_lock:
#             if latest_angles is None:
#                 time.sleep(0.001)
#                 continue
#             angles_new = latest_angles.copy()
#
#         # 插值
#         if prev_angles is None:
#             angles_display = angles_new
#         else:
#             alpha = 0.1
#             angles_display = (1 - alpha) * prev_angles + alpha * angles_new
#
#         # 计算
#         new_p_2, new_r_2 = update_robot_joints_1(robot_1, angles_display)
#         if new_p_2 is None or new_r_2 is None:
#             continue  # 跳过本次更新
#
#         fr3_conf = update_robot_joints_2(new_p_2, new_r_2)
#         if fr3_conf is None:
#             continue
#
#         with fr3_target_lock:
#             fr3_conf_new = fr3_conf.copy()
#         if not first_target_event.is_set():
#             first_target_event.set()
#
#         prev_fr3_conf = fr3_conf
#         prev_angles = angles_display.copy()
#
#         elapsed = time.time() - loop_start
#         sleep_time = 0.01 - elapsed
#         if sleep_time > 0:
#             time.sleep(sleep_time)
#         else:
#             print(f"警告：update_task 耗时 {elapsed:.4f} 秒")

def update_task_loop():
    global prev_angles, prev_fr3_conf, fr3_conf_new, plan_count
    while True:
        with count_lock:
            plan_count += 1
        loop_start = time.time()
        t_wait_start = loop_start

        if not T_ready.is_set():
            time.sleep(0.001)
            continue

        t_data_start = time.time()
        with data_lock:
            if latest_angles is None:
                time.sleep(0.001)
                continue
            angles_new = latest_angles.copy()
        t_data_end = time.time()

        # 插值
        if prev_angles is None:
            angles_display = angles_new
        else:
            alpha = 0.1
            angles_display = (1 - alpha) * prev_angles + alpha * angles_new
        t_interp_end = time.time()

        # 计算关节1
        new_p_2, new_r_2 = update_robot_joints_1(robot_1, angles_display)
        if new_p_2 is None or new_r_2 is None:
            continue
        t_joints1_end = time.time()

        # 计算关节2 (IK)
        fr3_conf = update_robot_joints_2(new_p_2, new_r_2)
        if fr3_conf is None:
            continue
        t_joints2_end = time.time()

        with fr3_target_lock:
            fr3_conf_new = fr3_conf.copy()
        if not first_target_event.is_set():
            first_target_event.set()
        prev_fr3_conf = fr3_conf
        prev_angles = angles_display.copy()
        t_end = time.time()

        elapsed = t_end - loop_start

        # 如果总耗时超过 0.02 秒，打印详细分解
        if elapsed > 0.02:
            print(f"plan_count={plan_count}, 总耗时={elapsed:.4f}s, 分解: "
                  f"T_ready等待={t_data_start-t_wait_start:.4f}, "
                  f"data_lock={t_data_end-t_data_start:.4f}, "
                  f"插值={t_interp_end-t_data_end:.4f}, "
                  f"joints1={t_joints1_end-t_interp_end:.4f}, "
                  f"joints2={t_joints2_end-t_joints1_end:.4f}, "
                  f"复制设置={t_end-t_joints2_end:.4f}")

        # 调整周期为 0.02 秒（原为 0.01）
        sleep_time = 0.02 - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # 超时警告已在上面打印，这里不再重复
            pass

# ================ 启动线程 =================
control_thread = threading.Thread(target=franka_control_thread, daemon=True)
control_thread.start()

plan_thread = threading.Thread(target=update_task_loop, daemon=True)
plan_thread.start()

# 主线程保持运行
last_print = time.time()
try:
    while True:
        time.sleep(0.5)
        now = time.time()
        if now - last_print >= 5:
            with count_lock:
                print(f"控制线程执行次数: {control_count}, 映射线程执行次数: {plan_count}")
            last_print = now
except KeyboardInterrupt:
    print("程序退出")
    server.close()