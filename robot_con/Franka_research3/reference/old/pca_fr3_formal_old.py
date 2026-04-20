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
from pylibfranka import Robot


def make_homo(rotmat, tvec):
    homo = np.eye(4)
    homo[:3, :3] = rotmat
    homo[:3, 3] = tvec
    return homo


# =============== 全局共享变量与锁 ===============
# 1. UDP 数据锁
first_angles = None                         # 第一次收到的主端数据
latest_angles = None                        # 此后发来的最新数据
prev_angles = None                          # 用于主端滤波
data_lock = threading.Lock()                # 用于在udp线程里更新数据时锁住
first_angles_received = threading.Event()   # 等待第一组主端数据来

# 2. IK 计算结果锁
fr3_target_lock = threading.Lock()
fr3_conf_new = None             # 存储最新的 IK 解算结果
T_ready = threading.Event()     # 等待转化关系计算成功
is_running = True               # 全局退出标志

# 3. 主端位姿转化为从端位姿的转化关系
T = None

# 4. udp接收主端和udp发送从端
UDP_IP = '0.0.0.0'
UDP_PORT = 14000
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((UDP_IP, UDP_PORT))
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
print("UDP 服务器已启动，等待数据...")

SEND_IP = '127.0.0.1'
SEND_PORT = 11113
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr = (SEND_IP, SEND_PORT)


#    ===============线程1：接收主端数据===============
def udp_receiver():
    global latest_angles, latest_time, first_angles
    while True:
        try:
            data, addr = server.recvfrom(1024)
            msg = np.frombuffer(data, dtype=np.float64, count=7)
            # print("接收到角度：", msg)

            angles = np.array(msg)
            angles = np.deg2rad(angles)
            # print("转为弧度：", angles)

            with data_lock:                                             # 收数据时，别的线程不能来读
                latest_angles = angles

            if first_angles is None:
                first_angles = angles.copy()
                first_angles_received.set()
                print("第一次收到的角度（rad）：", first_angles)

        except Exception as e:
            print(f"UDP 接收错误: {e}")

recv_thread = threading.Thread(target=udp_receiver, daemon=True)        # 线程1
recv_thread.start()


#    ===============获取主端位姿和从端位姿的转化关系===============
# base = wd.World(cam_pos=[-3, 2, 1], lookat_pos=[1, 0, 0])
# gm.gen_frame().attach_to(base)

robot_1 = pca.Pca()
first_angles_received.wait()
print("第一组数据到来")
robot_1.fk('arm', jnt_values=first_angles)
p_1, r_1 = robot_1.get_gl_tcp('arm')
init_1 = make_homo(r_1, p_1)           # 获取主端初始位姿
current_robot_mesh_1 = robot_1.gen_meshmodel(toggle_tcpcs=True)
# current_robot_mesh_1.attach_to(base)

robot_2 = fr3.Franka_research3()
fr3_pos = np.array([.50682, .6, 0])
fr3_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi)
robot_2.fix_to(fr3_pos, fr3_rot)

# initial_conf = get_initial_robot_position(ip="172.16.0.2")
initial_conf = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0])
print("机器人初始位置：", initial_conf)

robot_2.fk("arm", np.array(initial_conf))
p_2, r_2 = robot_2.get_gl_tcp('arm')
init_2 = make_homo(r_2, p_2)          # 获取从端初始位姿
current_robot_mesh_2 = robot_2.gen_meshmodel(toggle_tcpcs=True)
# current_robot_mesh_2.attach_to(base)

T = init_2 @ np.linalg.inv(init_1)    # 获得主端位姿和从端位姿的转化关系
T_ready.set()
print("主端到从端的转换关系是：", T)


#    ===============更新仿真环境里的主端和从端模型，并做运动学转化===============
def update_robot_joints_1(robot, jnt_values):
    global current_robot_mesh_1
    try:
        # if current_robot_mesh_1 is not None:
        #     current_robot_mesh_1.detach()
        robot.fk(jnt_values=jnt_values)
        p, r = robot.get_gl_tcp('arm')
        new_2 = T @ make_homo(r, p)           # 主端位姿转从端位姿

        new_p, new_r = new_2[:3, 3], new_2[:3, :3]
        # new_mesh = robot.gen_meshmodel()     # 更新主端仿真模型
        # new_mesh.attach_to(base)
        # current_robot_mesh_1 = new_mesh
        return new_p, new_r
    except Exception as e:
        print(f"更新主端机器人失败: {e}")

def update_robot_joints_2(pos, rotmat):
    global current_robot_mesh_2
    try:
        start_time = time.time()
        conf = robot_2.ik(component_name='arm', tgt_pos=pos, tgt_rotmat=rotmat)
        task_time = time.time() - start_time
        robot_2.fk(jnt_values=conf)
        if conf is None or (isinstance(conf, np.ndarray) and conf.size == 0):
            print("franka机器人无法解ik")

        # if current_robot_mesh_2 is not None:
        #     current_robot_mesh_2.detach()
        robot_2.fk(jnt_values=conf)

        # new_mesh = robot_2.gen_meshmodel()
        # new_mesh.attach_to(base)
        # current_robot_mesh_2 = new_mesh
        return conf, task_time
    except Exception as e:
        print(f"更新从端机器人时出错: {e}")
        return False


#    ===============主线程2：计算映射后的从端关节位置===============
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

        new_p_2, new_r_2 = update_robot_joints_1(robot_1, angles_display)
        if new_p_2 is None or new_r_2 is None:
            time.sleep(0.001)
            continue
        fr3_conf, ik_time = update_robot_joints_2(new_p_2, new_r_2)   # 得到从端conf
        if fr3_conf is None:
            time.sleep(0.001)
            continue

        with fr3_target_lock:
            fr3_conf_new = fr3_conf.copy()    # 更新从端conf

        prev_angles = angles_display.copy()

        # elapsed = time.time() - loop_start
        # print(f"update_task 耗时 {elapsed:.4f} 秒，其中解ik耗时 {ik_time:.4f} 秒")

        try:
            msg_bytes = struct.pack('7f', *fr3_conf_new.astype(np.float32))  # 转 float32 发送
            sender.sendto(msg_bytes, server_addr)
            time.sleep(0.001)
            # print("发送关节角度到franka控制端:", fr3_conf_new)
            print("\n")
        except Exception as e:
            print(f"发送失败: {e}")

plan_thread = threading.Thread(target=update_task_loop, daemon=True)
plan_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("程序退出")
    server.close()