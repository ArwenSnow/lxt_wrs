import sys
sys.path.insert(0, '/home/lruiqing/wrs_shu')
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
import struct
from scipy.spatial.transform import Rotation as Rot
from pynput import keyboard


def make_homo(rotmat, tvec):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo

# ===================按下空格切换===============================
# 程序退出标志
exit_flag = False

def on_press(key):
    global exit_flag
    if key == keyboard.Key.space:
        print("空格键按下，准备退出程序...")
        exit_flag = True
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

# =============== 全局共享变量与锁 ===============
# 1. UDP 数据锁
first_angles = None                         # 第一次收到的主端数据
latest_angles = None                        # 此后发来的最新数据
prev_angles = None                          # 用于主端滤波
data_lock = threading.Lock()                # 用于在udp线程里更新数据时锁住
first_angles_received = threading.Event()   # 等待第一组主端数据来

# 2. 映射计算结果锁
fr3_target_lock = threading.Lock()
fr3_conf_new = None             # 存储最新的 IK 解算结果
T_ready = threading.Event()     #
is_running = True               # 全局退出标志

# 3. 主端位姿转化为从端位姿的转化关系
T = None

# 4. 缩放因子
mapping_scale = 2

# 5. udp接收主端和udp发送从端,udp接收从端
UDP_IP = '0.0.0.0'
UDP_PORT = 14000
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
# print("UDP 服务器已启动，等待数据...")

SEND_IP = '127.0.0.1'
SEND_PORT = 11113
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr = (SEND_IP, SEND_PORT)


#    ======================线程1：接收主端数据=======================
def udp_receiver():
    global latest_angles, latest_time, first_angles
    while True:
        try:
            time_a = time.perf_counter()
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)
            # print("接收到角度：", msg)

            angles = np.array(msg)
            angles = np.deg2rad(angles)
            # print("转为弧度：", angles)

            with data_lock:                                             # 收数据时，别的线程不能来读
                latest_angles = angles
                time_b = time.perf_counter()
                # print("接收时间：", time_b - time_a)

            if first_angles is None:
                first_angles = angles.copy()
                first_angles_received.set()
                # print("第一次收到的角度（rad）：", first_angles)

        except Exception as e:
            print(f"UDP 接收错误: {e}")


recv_thread = threading.Thread(target=udp_receiver, daemon=True)        # 线程1
recv_thread.start()


#    ====================线程2：从从端那里接收初始位置===================
UDP_IP_2 = '0.0.0.0'
UDP_PORT_2 = 11114
server_2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_2.bind((UDP_IP_2, UDP_PORT_2))
server_2.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

init_conf = None
initconf_lock = threading.Lock()
def udp_receiver_2():
    global init_conf
    while True:
        try:
            data, addr = server_2.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)
            with initconf_lock:
                init_conf = msg
        except Exception as e:
            print(f"UDP 接收错误: {e}")

recv_thread_2 = threading.Thread(target=udp_receiver_2, daemon=True)
recv_thread_2.start()


#    =================获取主端位姿和从端位姿的转化关系====================
robot_1 = pca.Pca()
first_angles_received.wait()
# print("第一组数据到来")
robot_1.fk('arm', jnt_values=first_angles)
p_1, r_1 = robot_1.get_gl_tcp('arm')    # 主端初始位姿

robot_2 = fr3.Franka_research3()
fr3_pos = np.array([.6, 0, 0])
fr3_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi/2)
robot_2.fix_to(fr3_pos, fr3_rot)

robot_2.fk("arm", np.array(init_conf))
p_2, r_2 = robot_2.get_gl_tcp('arm')    # 从端初始位姿

rot_delta = r_2 @ np.linalg.inv(r_1)
T_ready.set()


#    ===============更新仿真环境里的主端和从端模型，并做运动学转化===============
def update_robot_joints_1(robot, jnt_values):
    global current_robot_mesh_1, p_1, r_1, p_2, r_2
    try:
        robot.fk(jnt_values=jnt_values)
        p, r = robot.get_gl_tcp('arm')
        new_p = p_2 + rot_delta @ ((p - p_1) * mapping_scale)

        # rot_rel = Rot.from_matrix(r_1.T @ r)    # 创建一个rotation对象
        # angle = rot_rel.magnitude()             # 旋转的角度（弧度）
        # if angle < 1e-12:
        #     rot_scaled = np.eye(3)
        # else:
        #     axis = rot_rel.as_rotvec() / angle  # 旋转轴 = 旋转的向量/旋转的角度（弧度）
        #     rot_scaled = Rot.from_rotvec(axis * (angle * mapping_scale)).as_matrix()
        # new_r = r_2 @ rot_scaled
        new_r = r_2 @ (r_1.T @ r)

        return new_p, new_r
    except Exception as e:
        print(f"更新主端机器人失败: {e}")


def update_robot_joints_2(pos, rotmat):
    global current_robot_mesh_2
    try:
        start_time = time.time()
        current_jnts = robot_2.get_jnt_values("arm")
        pos_r = fr3_rot.T.dot(pos - fr3_pos)
        rotmat_r = fr3_rot.T.dot(rotmat)
        conf = robot_2.tracik(tgt_pos=pos_r, tgt_rotmat=rotmat_r, seed_jnt_values=current_jnts)
        if conf is None:
            print("tracik解不出")
            conf = robot_2.ik(component_name='arm', tgt_pos=pos, tgt_rotmat=rotmat)
        task_time = time.time() - start_time
        robot_2.fk(jnt_values=conf)
        if conf is None or (isinstance(conf, np.ndarray) and conf.size == 0):
            print("franka机器人无法解ik")
            return None, task_time

        robot_2.fk(jnt_values=conf)

        return conf, task_time
    except Exception as e:
        print(f"更新从端机器人时出错: {e}")
        return False


#    ===============线程3：计算映射后的从端关节位置===============
def update_task_loop():
    global prev_angles, fr3_conf_new
    while True:
        # loop_start = time.time()
        if not T_ready.is_set():
            time.sleep(0.001)
            continue
        with data_lock:
            if latest_angles is None:
                time.sleep(0.001)
                continue
            angles_new = latest_angles.copy()    # 更新收到的主端数据

        if prev_angles is None:
            angles_display = angles_new
        else:
            alpha = 0.1
            angles_display = (1 - alpha) * prev_angles + alpha * angles_new   # 插值，让主端关节位置变化更平滑

        time_1 = time.perf_counter()
        new_p_2, new_r_2 = update_robot_joints_1(robot_1, angles_display)
        time_2 = time.perf_counter()
        if new_p_2 is None or new_r_2 is None:
            time.sleep(0.001)
            continue
        time_3 = time.perf_counter()
        fr3_conf, ik_time = update_robot_joints_2(new_p_2, new_r_2)   # 得到从端conf
        time_4 = time.perf_counter()
        if fr3_conf is None:
            time.sleep(0.001)
            continue

        with fr3_target_lock:
            fr3_conf_new = fr3_conf.copy()    # 更新从端conf

        prev_angles = angles_display.copy()

        # elapsed = time.time() - loop_start
        # print(f"update_task 耗时 {elapsed:.4f} 秒，其中解ik耗时 {ik_time:.4f} 秒")

        time_5 = time.perf_counter()
        try:
            msg_bytes = struct.pack('7f', *fr3_conf_new.astype(np.float32))  # 转 float32 发送
            sender.sendto(msg_bytes, server_addr)
            time_6 = time.perf_counter()
            # print("步骤1：",time_2 - time_1, "解ik：", time_4 - time_3, "发送：", time_6 - time_5)
            time.sleep(0.001)
            # print("发送关节角度到franka控制端:", fr3_conf_new)
            print("\n")
        except Exception as e:
            print(f"发送失败: {e}")


plan_thread = threading.Thread(target=update_task_loop, daemon=True)
plan_thread.start()

try:
    while not exit_flag:
        time.sleep(1)
except KeyboardInterrupt:
    exit_flag = True

listener.stop()
server.close()
server_2.close()